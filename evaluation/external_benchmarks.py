"""Normalize, sample, and align external scientific-QA benchmarks."""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from generation.citation_handler import lexical_overlap_score
except ImportError:
    from ..generation.citation_handler import lexical_overlap_score


_MULTIMODAL = re.compile(
    r"\b(?:figure|fig\.?|table|equation|eq\.?|appendix|supplement(?:ary)?|image|plot|chart)\b",
    re.IGNORECASE,
)
_MULTIDOC = re.compile(
    r"\b(?:cited paper|reference \[|prior work|other paper|second paper|bibliograph)\w*",
    re.IGNORECASE,
)
_REASONING = re.compile(
    r"\b(?:how|why|compare|comparison|differ|difference|reason|rationale|explain|implication)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ExternalExample:
    source_dataset: str
    source_id: str
    question: str
    reference_answer: str
    evidence: list[str]
    paper_id: str = ""
    title: str = ""
    answer_type: str = "free_form"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AlignmentResult:
    chunk_ids: list[str]
    reviewed: bool
    reason: str | None
    scores: list[float] = field(default_factory=list)


@dataclass(slots=True)
class TierReport:
    source_dataset: str
    available: int
    sampled: int
    aligned: int
    skipped: dict[str, int]
    output_path: str

    @property
    def alignment_rate(self) -> float:
        return self.aligned / self.sampled if self.sampled else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "alignment_rate": self.alignment_rate}


class ExternalBenchmarkBuilder:
    """Dependency-injected orchestration for network-free tests and live builds."""

    def __init__(
        self,
        *,
        loaders: Mapping[str, Callable[[], Any]],
        chunks: Sequence[Mapping[str, Any]],
        paper_preparer: (
            Callable[[ExternalExample], Sequence[Mapping[str, Any]]] | None
        ) = None,
    ) -> None:
        self.loaders = dict(loaders)
        self.chunks = [dict(chunk) for chunk in chunks]
        self.paper_preparer = paper_preparer

    def build(
        self,
        output_dir: str | Path,
        *,
        qasa_cap: int = 60,
        qasper_cap: int = 30,
        scidqa_cap: int = 15,
        seeds: tuple[int, int, int] = (1701, 2701, 3701),
        threshold: float = 0.55,
        ambiguity_margin: float = 0.05,
    ) -> dict[str, Any]:
        normalized = {
            "qasa": normalize_qasa(self.loaders["qasa"]()),
            "qasper": normalize_qasper(self.loaders["qasper"]()),
            "scidqa": normalize_scidqa(self.loaders["scidqa"]()),
        }
        scidqa_sample, scidqa_skips = sample_scidqa(
            normalized["scidqa"], scidqa_cap, seeds[2]
        )
        samples = {
            "qasa": sample_qasa(normalized["qasa"], qasa_cap, seeds[0]),
            "qasper": sample_qasper(normalized["qasper"], qasper_cap, seeds[1]),
            "scidqa": scidqa_sample,
        }
        grouped = chunks_by_paper(self.chunks)
        preparation_failures: set[str] = set()
        if self.paper_preparer is not None:
            for row in {
                f"{item.source_dataset}:{item.source_id}": item
                for values in samples.values()
                for item in values
            }.values():
                if _lookup_chunks(grouped, row):
                    continue
                try:
                    new_chunks = [dict(chunk) for chunk in self.paper_preparer(row)]
                except Exception:
                    new_chunks = []
                if new_chunks:
                    self.chunks.extend(new_chunks)
                    grouped = chunks_by_paper(self.chunks)
                else:
                    preparation_failures.add(
                        f"{row.source_dataset}:{row.source_id}"
                    )
        destination = Path(output_dir)
        reports: dict[str, TierReport] = {}
        for name, rows in samples.items():
            initial: Counter[str] = Counter(scidqa_skips if name == "scidqa" else {})
            report = build_tier(
                rows,
                grouped,
                output_path=destination / f"{name}_generation_qa.json",
                source_dataset=name,
                initial_skips=initial,
                preparation_failures=preparation_failures,
                threshold=threshold,
                ambiguity_margin=ambiguity_margin,
            )
            report.available = len(normalized[name])
            reports[name] = report
        result = {
            "tiers": {name: report.to_dict() for name, report in reports.items()},
            "total_records": sum(report.sampled for report in reports.values()),
            "total_aligned": sum(report.aligned for report in reports.values()),
        }
        _atomic_json(destination / "external_benchmark_report.json", result)
        return result


def normalize_qasa(payload: Any) -> list[ExternalExample]:
    """Normalize the published QASA v1/v1.1 JSON schema."""

    rows = list(payload.values()) if isinstance(payload, Mapping) else list(payload)
    result: list[ExternalExample] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        evidential = row.get("evidential_info") or []
        evidence = [
            _text(item.get("context"))
            for item in evidential
            if isinstance(item, Mapping) and _text(item.get("context"))
        ]
        identifier = _text(row.get("question_id") or row.get("id"))
        paper_id = canonical_paper_id(row.get("arxiv_id") or row.get("paper_id"))
        question, answer = _text(row.get("question")), _text(
            row.get("composition") or row.get("answer")
        )
        if identifier and question and answer:
            result.append(
                ExternalExample(
                    "qasa",
                    f"{paper_id}-{identifier}",
                    question,
                    answer,
                    evidence,
                    paper_id,
                    _text(row.get("title")),
                    metadata={
                        "question_type": _text(row.get("question_type")),
                        "sampling_id": identifier,
                    },
                )
            )
    return result


def normalize_qasper(payload: Any) -> list[ExternalExample]:
    """Flatten QASPER documents and their nested answer annotations."""

    documents = (
        list(payload.values()) if isinstance(payload, Mapping) else list(payload)
    )
    result: list[ExternalExample] = []
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        qas = document.get("qas") or {}
        if isinstance(qas, Mapping):
            questions = _as_list(qas.get("question"))
            ids = _as_list(qas.get("question_id"))
            answers = _as_list(qas.get("answers"))
        else:
            qa_rows = [item for item in _as_list(qas) if isinstance(item, Mapping)]
            questions = [item.get("question") for item in qa_rows]
            ids = [item.get("question_id") for item in qa_rows]
            answers = [item.get("answers") for item in qa_rows]
        for index, question in enumerate(questions):
            answer = _first_answer(answers[index] if index < len(answers) else None)
            if not answer:
                continue
            answer_text, answer_type = _qasper_answer(answer)
            if not answer_text:
                continue
            evidence = _strings(
                answer.get("highlighted_evidence") or answer.get("evidence") or []
            )
            identifier = _text(ids[index] if index < len(ids) else index)
            result.append(
                ExternalExample(
                    "qasper",
                    identifier,
                    _text(question),
                    answer_text,
                    evidence,
                    canonical_paper_id(
                        document.get("arxiv_id")
                        or document.get("paper_id")
                        or document.get("id")
                    ),
                    _text(document.get("title")),
                    answer_type,
                )
            )
    return [row for row in result if row.source_id and row.question]


def normalize_scidqa(payload: Any) -> list[ExternalExample]:
    """Normalize SciDQA, retaining evidence only when an export actually supplies it."""

    rows = list(payload.values()) if isinstance(payload, Mapping) else list(payload)
    result: list[ExternalExample] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        identifier = _text(row.get("id") or row.get("qid"))
        question = _text(row.get("que") or row.get("question"))
        answer = _text(row.get("ans") or row.get("answer"))
        evidence = _strings(
            row.get("evidence")
            or row.get("evidence_text")
            or row.get("supporting_passages")
            or row.get("source_evidence")
            or []
        )
        if identifier and question and answer:
            result.append(
                ExternalExample(
                    "scidqa",
                    identifier,
                    question,
                    re.sub(r"^A:\s*", "", answer).strip(),
                    evidence,
                    canonical_paper_id(
                        row.get("arxiv_id") or row.get("paper_id") or row.get("pid")
                    ),
                    _text(row.get("title")),
                    metadata={
                        "version": _text(row.get("version")),
                        "rid": _text(row.get("rid")),
                    },
                )
            )
    return result


def sample_qasa(
    examples: Sequence[ExternalExample], cap: int = 60, seed: int = 1701
) -> list[ExternalExample]:
    preferred = [row for row in examples if _REASONING.search(row.question)]
    remainder = [row for row in examples if row not in preferred]
    return _seeded_take(preferred, cap, seed) + _seeded_take(
        remainder, max(0, cap - min(cap, len(preferred))), seed + 1
    )


def sample_qasper(
    examples: Sequence[ExternalExample], cap: int = 30, seed: int = 2701
) -> list[ExternalExample]:
    """Deterministically preserve extractive, yes/no, unanswerable, and free-form mix."""

    groups: dict[str, list[ExternalExample]] = defaultdict(list)
    for row in examples:
        groups[row.answer_type].append(row)
    order = ("extractive", "yes_no", "unanswerable", "free_form")
    selected: list[ExternalExample] = []
    shuffled = {
        name: _seeded_take(groups[name], len(groups[name]), seed + i)
        for i, name in enumerate(order)
    }
    while len(selected) < cap and any(shuffled.values()):
        for name in order:
            if shuffled[name] and len(selected) < cap:
                selected.append(shuffled[name].pop())
    return selected


def sample_scidqa(
    examples: Sequence[ExternalExample], cap: int = 15, seed: int = 3701
) -> tuple[list[ExternalExample], dict[str, int]]:
    eligible: list[ExternalExample] = []
    skipped: Counter[str] = Counter()
    for row in examples:
        combined = " ".join([row.question, row.reference_answer, *row.evidence])
        if not row.evidence:
            skipped["missing_evidence"] += 1
        elif _MULTIMODAL.search(combined):
            skipped["multimodal"] += 1
        elif _MULTIDOC.search(combined):
            skipped["multidocument"] += 1
        else:
            eligible.append(row)
    return _seeded_take(eligible, cap, seed), dict(skipped)


def align_evidence_to_chunks(
    evidence: Sequence[str],
    chunks: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.55,
    ambiguity_margin: float = 0.05,
) -> AlignmentResult:
    """Map evidence text to deterministic project chunks using citation overlap semantics."""

    if not chunks:
        return AlignmentResult([], False, "no_chunks")
    if not evidence:
        return AlignmentResult([], False, "missing_evidence")
    selected: list[str] = []
    best_scores: list[float] = []
    for paragraph in evidence:
        ranked = sorted(
            (
                (lexical_overlap_score(paragraph, _text(chunk.get("text"))), chunk)
                for chunk in chunks
                if _text(chunk.get("chunk_id")) and _text(chunk.get("text"))
            ),
            key=lambda item: (-item[0], _text(item[1].get("chunk_id"))),
        )
        if not ranked or ranked[0][0] < threshold:
            return AlignmentResult(selected, False, "below_threshold", best_scores)
        best_score, best = ranked[0]
        competitors = [
            item for item in ranked[1:] if best_score - item[0] <= ambiguity_margin
        ]
        if competitors and any(not _adjacent(best, item[1]) for item in competitors):
            return AlignmentResult(
                selected, False, "ambiguous", [*best_scores, best_score]
            )
        paragraph_chunks = [
            best,
            *(item[1] for item in competitors if _adjacent(best, item[1])),
        ]
        combined = " ".join(_text(chunk.get("text")) for chunk in paragraph_chunks)
        if lexical_overlap_score(paragraph, combined) < threshold:
            return AlignmentResult(selected, False, "below_threshold", best_scores)
        selected.extend(_text(chunk.get("chunk_id")) for chunk in paragraph_chunks)
        best_scores.append(best_score)
    return AlignmentResult(list(dict.fromkeys(selected)), True, None, best_scores)


def build_tier(
    examples: Sequence[ExternalExample],
    chunks_by_paper: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_path: str | Path,
    source_dataset: str | None = None,
    initial_skips: Mapping[str, int] | None = None,
    preparation_failures: set[str] | None = None,
    threshold: float = 0.55,
    ambiguity_margin: float = 0.05,
) -> TierReport:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter(initial_skips or {})
    aligned = 0
    for row in examples:
        chunks = _lookup_chunks(chunks_by_paper, row)
        preparation_key = f"{row.source_dataset}:{row.source_id}"
        if not chunks and preparation_key in (preparation_failures or set()):
            alignment = AlignmentResult([], False, "ingestion_failure")
        else:
            alignment = align_evidence_to_chunks(
                row.evidence,
                chunks,
                threshold=threshold,
                ambiguity_margin=ambiguity_margin,
            )
        if alignment.reviewed:
            aligned += 1
        else:
            skipped[alignment.reason or "alignment_failure"] += 1
        records.append(
            {
                "id": f"{row.source_dataset}-{row.source_id}",
                "question": row.question,
                "reference_answer": row.reference_answer,
                "reference_context_ids": alignment.chunk_ids,
                "retrieved_chunk_ids": [],
                "reviewed": alignment.reviewed,
                "source_dataset": row.source_dataset,
            }
        )
    destination = Path(output_path)
    _atomic_json(destination, {"schema_version": 1, "questions": records})
    source = source_dataset or (
        examples[0].source_dataset if examples else destination.stem
    )
    return TierReport(
        source, len(examples), len(records), aligned, dict(skipped), str(destination)
    )


def chunks_by_paper(
    chunks: Iterable[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        record = dict(chunk)
        grouped[canonical_paper_id(chunk.get("paper_id"))].append(record)
        metadata = chunk.get("metadata")
        title = _text(metadata.get("title")) if isinstance(metadata, Mapping) else ""
        if title:
            grouped[f"title:{_normalized_title(title)}"].append(record)
    return dict(grouped)


def canonical_paper_id(value: Any) -> str:
    text = _text(value).rstrip("/").split("/")[-1].casefold()
    return re.sub(r"v\d+$", "", text)


def _lookup_chunks(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]], row: ExternalExample
) -> list[Mapping[str, Any]]:
    chunks = list(grouped.get(canonical_paper_id(row.paper_id), []))
    if not chunks and row.title:
        chunks = list(grouped.get(f"title:{_normalized_title(row.title)}", []))
    return chunks


def _normalized_title(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _text(value).casefold()))


def load_json_records(path: str | Path) -> Any:
    source = Path(path)
    if source.suffix.casefold() == ".jsonl":
        return [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return json.loads(source.read_text(encoding="utf-8"))


def _qasper_answer(answer: Mapping[str, Any]) -> tuple[str, str]:
    if bool(answer.get("unanswerable")):
        return "The question is unanswerable from the paper.", "unanswerable"
    spans = _strings(answer.get("extractive_spans") or [])
    if spans:
        return " ".join(spans), "extractive"
    yes_no = answer.get("yes_no")
    if isinstance(yes_no, bool):
        return "Yes." if yes_no else "No.", "yes_no"
    return _text(answer.get("free_form_answer")), "free_form"


def _first_answer(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        nested = value.get("answer")
        if nested is not None:
            return _first_answer(nested)
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            answer = _first_answer(item)
            if answer is not None:
                return answer
    return None


def _as_list(value: Any) -> list[Any]:
    return (
        list(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else []
    )


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Sequence):
        return []
    return list(dict.fromkeys(_text(item) for item in value if _text(item)))


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _seeded_take(
    examples: Sequence[ExternalExample], cap: int, seed: int
) -> list[ExternalExample]:
    rows = sorted(
        examples,
        key=lambda row: (row.metadata.get("sampling_id", row.source_id), row.question),
    )
    random.Random(seed).shuffle(rows)
    return rows[: max(0, cap)]


def _adjacent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if _text(left.get("section")) != _text(right.get("section")):
        return False
    try:
        left_start, left_end = int(left["start_char"]), int(left["end_char"])
        right_start, right_end = int(right["start_char"]), int(right["end_char"])
    except (KeyError, TypeError, ValueError):
        return False
    return min(abs(left_end - right_start), abs(right_end - left_start)) <= 1 or not (
        left_end < right_start or right_end < left_start
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
