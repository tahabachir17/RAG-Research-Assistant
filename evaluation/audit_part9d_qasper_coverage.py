"""Audit whether zero-Recall@4 QASPER evidence exists in the BM25 corpus."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from .external_benchmarks import ExternalExample, normalize_qasper
except ImportError:
    from external_benchmarks import ExternalExample, normalize_qasper

from generation.citation_handler import lexical_overlap_score
from processing.bm25_indexer import BM25Indexer


@dataclass(slots=True)
class CoverageFinding:
    question_id: str
    question: str
    bucket: str
    evidence_count: int
    gold_chunk_count: int
    gold_chunks_in_index: int
    paper_chunks_in_index: int
    exact_substring: bool
    best_overlap: float
    best_chunk_id: str
    best_section: str
    cause: str


class QasperCoverageAuditor:
    """Classify zero-recall rows against injected benchmark artifacts."""

    def __init__(
        self,
        index: BM25Indexer,
        examples: Sequence[ExternalExample],
        golden_by_id: dict[str, dict[str, Any]],
        *,
        overlap_threshold: float = 0.55,
    ) -> None:
        self.index = index
        self.examples = {f"qasper-{row.source_id}": row for row in examples}
        self.golden_by_id = golden_by_id
        self.overlap_threshold = overlap_threshold
        self.chunk_by_id = {
            str(chunk.get("chunk_id")): chunk for chunk in self.index.chunks
        }

    def audit(self, score_rows: Sequence[dict[str, Any]]) -> list[CoverageFinding]:
        zero_rows = [
            row
            for row in score_rows
            if row.get("tier") == "qasper" and float(row.get("recall@4", 0.0)) == 0.0
        ]
        return [self._finding(row) for row in zero_rows]

    def _finding(self, score_row: dict[str, Any]) -> CoverageFinding:
        question_id = str(score_row["query_id"])
        example = self.examples[question_id]
        golden = self.golden_by_id[question_id]
        gold_ids = [str(item) for item in golden.get("reference_context_ids", [])]
        gold_chunks = [self.chunk_by_id[item] for item in gold_ids if item in self.chunk_by_id]
        paper_chunks = [
            chunk
            for chunk in self.index.chunks
            if str(chunk.get("paper_id", "")) == example.paper_id
        ]
        best_score, best_chunk = self._best_match(example.evidence)
        exact = any(
            _normalized(evidence) in _normalized(str(chunk.get("text", "")))
            for evidence in example.evidence
            for chunk in self.index.chunks
            if _normalized(evidence)
        )
        present = bool(gold_chunks) and (
            exact
            or best_score >= self.overlap_threshold
            or self._gold_combined_overlap(example.evidence, gold_chunks)
            >= self.overlap_threshold
        )
        bucket = "(b) Evidence present, not retrieved" if present else "(a) Evidence absent"
        cause = self._cause(example, gold_chunks, paper_chunks, best_chunk, present)
        return CoverageFinding(
            question_id=question_id,
            question=str(score_row["question"]),
            bucket=bucket,
            evidence_count=len(example.evidence),
            gold_chunk_count=len(gold_ids),
            gold_chunks_in_index=len(gold_chunks),
            paper_chunks_in_index=len(paper_chunks),
            exact_substring=exact,
            best_overlap=best_score,
            best_chunk_id=str(best_chunk.get("chunk_id", "")) if best_chunk else "",
            best_section=str(best_chunk.get("section", "")) if best_chunk else "",
            cause=cause,
        )

    def _best_match(
        self, evidence: Sequence[str]
    ) -> tuple[float, dict[str, Any] | None]:
        best_score = 0.0
        best_chunk: dict[str, Any] | None = None
        for paragraph in evidence:
            for chunk in self.index.chunks:
                score = lexical_overlap_score(paragraph, str(chunk.get("text", "")))
                if score > best_score:
                    best_score, best_chunk = score, chunk
        return best_score, best_chunk

    @staticmethod
    def _gold_combined_overlap(
        evidence: Sequence[str], chunks: Sequence[dict[str, Any]]
    ) -> float:
        combined = " ".join(str(chunk.get("text", "")) for chunk in chunks)
        return max(
            (lexical_overlap_score(paragraph, combined) for paragraph in evidence),
            default=0.0,
        )

    @staticmethod
    def _cause(
        example: ExternalExample,
        gold_chunks: Sequence[dict[str, Any]],
        paper_chunks: Sequence[dict[str, Any]],
        best_chunk: dict[str, Any] | None,
        present: bool,
    ) -> str:
        if present:
            sections = sorted(
                {str(chunk.get("section", "")).strip() for chunk in gold_chunks}
            )
            return "Reviewed evidence is indexed in " + (
                ", ".join(section for section in sections if section) or "labeled chunks"
            ) + "; this is a top-4 ranking miss."
        if not paper_chunks:
            return "Source paper has no chunks in the benchmark BM25 index."
        combined = " ".join(example.evidence)
        if re.search(r"\b(?:table|figure|fig\.|caption)\b", combined, re.IGNORECASE):
            return "Source paper is indexed, but labeled evidence appears table/figure-related and is absent."
        section = str((best_chunk or {}).get("section", "")).strip()
        return (
            "Source paper is indexed, but no chunk reaches the evidence-overlap threshold"
            + (f"; nearest text is in {section}." if section else ".")
        )


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), re.UNICODE))


def _load_golden(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in payload["questions"]}


def write_outputs(findings: Sequence[CoverageFinding], report: Path, artifact: Path) -> None:
    absent = sum(row.bucket.startswith("(a)") for row in findings)
    present = len(findings) - absent
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "zero_recall_qasper": len(findings),
                "evidence_absent": absent,
                "evidence_present_not_retrieved": present,
                "findings": [asdict(row) for row in findings],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Part 9d — QASPER Evidence-Coverage Audit",
        "",
        "Date: 2026-08-13",
        "",
        f"Of {len(findings)} QASPER questions with Recall@4 = 0, **{absent}** are bucket (a) evidence absent and **{present}** are bucket (b) evidence present but not retrieved.",
        "",
        "| Question ID | Bucket | Indexed gold chunks | Best overlap | Evidence location / cause |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in findings:
        lines.append(
            f"| `{row.question_id}` | {row.bucket} | {row.gold_chunks_in_index}/{row.gold_chunk_count} | {row.best_overlap:.3f} | {row.cause} |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "The audit reconstructed original evidence from `evaluation/data/external_cache/qasper.json`, loaded all chunks from `external_bm25_index.pkl`, and checked normalized exact substrings plus the same lexical-overlap threshold (0.55) used by benchmark evidence alignment. Stored reviewed chunk IDs were independently checked for presence in the current BM25 corpus.",
            "",
            "This is a coverage diagnosis only; no retrieval or chunking logic was changed.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("evaluation/data/eval_results/part9c_retrieval_20260813/retrieval_scores.json"),
    )
    parser.add_argument(
        "--qasper-source",
        type=Path,
        default=Path("evaluation/data/external_cache/qasper.json"),
    )
    parser.add_argument(
        "--qasper-golden",
        type=Path,
        default=Path("evaluation/data/external_benchmarks/qasper_generation_qa.json"),
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=Path("evaluation/data/external_benchmarks/external_bm25_index.pkl"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/part9d_qasper_evidence_coverage.md")
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("evaluation/data/eval_results/part9d_20260813/qasper_evidence_coverage.json"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    scores = json.loads(args.scores.read_text(encoding="utf-8"))["per_question"]
    source = json.loads(args.qasper_source.read_text(encoding="utf-8"))
    auditor = QasperCoverageAuditor(
        BM25Indexer.load(args.bm25_index),
        normalize_qasper(source),
        _load_golden(args.qasper_golden),
    )
    findings = auditor.audit(scores)
    write_outputs(findings, args.report, args.artifact)
    print(json.dumps([asdict(row) for row in findings], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
