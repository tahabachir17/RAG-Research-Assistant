from evaluation.judge_reliability_diagnostic import anomaly_fixed, smoke_anomalies


def test_smoke_anomalies_selects_invalid_and_hard_zero_only():
    report = {
        "ragas": {
            "questions": [
                {
                    "id": "invalid",
                    "faithfulness": None,
                    "answer_relevancy": 0.4,
                    "reasons": {"faithfulness": "1 of 1 metric values were unavailable"},
                },
                {"id": "zero", "answer_relevancy": 0.0},
                {
                    "id": "dataset-gap",
                    "answer_relevancy": 0.5,
                    "reasons": {"context_recall": "gold evidence is unavailable"},
                },
            ]
        }
    }

    assert smoke_anomalies(report) == [
        {"question_id": "invalid", "metric": "faithfulness", "kind": "invalid_json"},
        {"question_id": "zero", "metric": "answer_relevancy", "kind": "hard_zero"},
    ]


def test_anomaly_fixed_requires_nonzero_for_suspicious_zero():
    assert anomaly_fixed("invalid_json", 0.0) is True
    assert anomaly_fixed("hard_zero", 0.0) is False
    assert anomaly_fixed("hard_zero", 0.2) is True
    assert anomaly_fixed("invalid_json", None) is False
