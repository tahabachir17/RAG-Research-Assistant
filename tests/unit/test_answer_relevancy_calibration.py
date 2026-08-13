from __future__ import annotations

from evaluation.answer_relevancy_calibration import (
    build_calibration_report,
    load_calibration_cases,
)
from evaluation.full_ragas_evaluation import metric_release_result


def test_persisted_calibration_compares_human_labels_with_ragas_scores():
    cases = load_calibration_cases(
        "evaluation/calibration/answer_relevancy_labels.json"
    )

    report = build_calibration_report(cases)

    assert 10 <= report["summary"]["cases"] <= 20
    assert report["summary"]["exact_label_agreement"] < 0.80
    assert report["summary"]["acceptable"] is False
    reddit = next(
        row for row in report["cases"] if row["question_id"] == "controlled-reddit-01"
    )
    assert reddit["human_label"] == "directly_relevant"
    assert reddit["ragas_answer_relevancy"] < 0.30


def test_answer_relevancy_is_diagnostic_only_for_release_decisions():
    assert metric_release_result("answer_relevancy", 0.01) == "diagnostic only"
    assert metric_release_result("faithfulness", 1.0) == "pass"
