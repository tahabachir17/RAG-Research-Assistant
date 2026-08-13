from __future__ import annotations

import json

from evaluation.external_benchmarks import (
    ExternalBenchmarkBuilder,
    ExternalExample,
    align_evidence_to_chunks,
    sample_qasa,
    sample_qasper,
    sample_scidqa,
)


def _example(source: str, identifier: str, **changes):
    values = {
        "source_dataset": source,
        "source_id": identifier,
        "question": f"Why does method {identifier} work?",
        "reference_answer": "Because it uses grounded evidence.",
        "evidence": ["The method uses grounded evidence to improve accuracy."],
        "paper_id": "2401.00001",
        "answer_type": "free_form",
    }
    values.update(changes)
    return ExternalExample(**values)


def test_clean_evidence_alignment_uses_project_chunk_id():
    result = align_evidence_to_chunks(
        ["The proposed encoder improves accuracy on the benchmark."],
        [
            {
                "chunk_id": "gold",
                "text": "The proposed encoder improves accuracy on the benchmark.",
                "section": "results",
                "start_char": 0,
                "end_char": 58,
            },
            {"chunk_id": "noise", "text": "Unrelated background material."},
        ],
    )
    assert result.reviewed is True
    assert result.chunk_ids == ["gold"]


def test_ambiguous_split_fails_instead_of_guessing():
    evidence = "alpha beta gamma delta epsilon zeta eta theta"
    result = align_evidence_to_chunks(
        [evidence],
        [
            {"chunk_id": "left", "text": "alpha beta gamma delta"},
            {"chunk_id": "right", "text": "epsilon zeta eta theta"},
        ],
        threshold=0.45,
    )
    assert result.reviewed is False
    assert result.reason == "ambiguous"


def test_alignment_reports_paper_not_ingested():
    result = align_evidence_to_chunks(["some evidence"], [])
    assert result.reviewed is False
    assert result.reason == "no_chunks"


def test_sampling_is_deterministic_and_qasper_preserves_types():
    qasa = [_example("qasa", str(index)) for index in range(20)]
    assert [row.source_id for row in sample_qasa(qasa, 8, 9)] == [
        row.source_id for row in sample_qasa(qasa, 8, 9)
    ]
    qasper = [
        _example("qasper", f"{kind}-{index}", answer_type=kind)
        for kind in ("extractive", "yes_no", "unanswerable", "free_form")
        for index in range(4)
    ]
    selected = sample_qasper(qasper, 8, 11)
    assert {row.answer_type for row in selected} == {
        "extractive",
        "yes_no",
        "unanswerable",
        "free_form",
    }
    assert [row.source_id for row in selected] == [
        row.source_id for row in sample_qasper(qasper, 8, 11)
    ]


def test_scidqa_excludes_missing_evidence_multimodal_and_multidocument():
    examples = [
        _example("scidqa", "ok"),
        _example("scidqa", "missing", evidence=[]),
        _example("scidqa", "figure", question="What is shown in Figure 2?"),
        _example("scidqa", "other", question="How does the cited paper differ?"),
    ]
    selected, skipped = sample_scidqa(examples, 15, 3)
    assert [row.source_id for row in selected] == ["ok"]
    assert skipped == {"missing_evidence": 1, "multimodal": 1, "multidocument": 1}


def test_builder_uses_injected_loaders_and_writes_separate_tiers(tmp_path):
    qasa = {
        "0": {
            "question_id": "1",
            "question": "Why does it work?",
            "composition": "It uses evidence.",
            "evidential_info": [{"context": "The method uses evidence."}],
            "arxiv_id": "2401.00001",
        }
    }
    qasper = []
    scidqa = []
    chunks = [
        {
            "chunk_id": "chunk-1",
            "paper_id": "2401.00001v2",
            "section": "method",
            "text": "The method uses evidence.",
            "start_char": 0,
            "end_char": 25,
            "metadata": {},
        }
    ]
    builder = ExternalBenchmarkBuilder(
        loaders={
            "qasa": lambda: qasa,
            "qasper": lambda: qasper,
            "scidqa": lambda: scidqa,
        },
        chunks=chunks,
        paper_preparer=lambda _: (_ for _ in ()).throw(AssertionError("not needed")),
    )
    report = builder.build(tmp_path, qasa_cap=1, qasper_cap=0, scidqa_cap=0)
    payload = json.loads((tmp_path / "qasa_generation_qa.json").read_text())
    assert report["total_records"] == 1
    assert payload["questions"][0]["reference_context_ids"] == ["chunk-1"]
    assert payload["questions"][0]["reviewed"] is True
