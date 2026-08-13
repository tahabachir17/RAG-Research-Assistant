"""Compare human relevance labels with RAGAS answer-relevancy scores."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LABEL_VALUES = {"irrelevant": 0.0, "partially_relevant": 0.5, "directly_relevant": 1.0}


@dataclass(slots=True)
class RelevancyCalibrationCase:
    question_id: str
    question: str
    answer: str
    human_label: str
    ragas_answer_relevancy: float


def load_calibration_cases(path: str | Path) -> list[RelevancyCalibrationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not 10 <= len(records) <= 20:
        raise ValueError("answer-relevancy calibration requires 10 to 20 cases")
    cases = [RelevancyCalibrationCase(**record) for record in records]
    if any(case.human_label not in LABEL_VALUES for case in cases):
        raise ValueError("human_label must be directly_relevant, partially_relevant, or irrelevant")
    return cases


def score_to_label(score: float) -> str:
    if score >= 2.0 / 3.0:
        return "directly_relevant"
    if score >= 1.0 / 3.0:
        return "partially_relevant"
    return "irrelevant"


def build_calibration_report(cases: list[RelevancyCalibrationCase]) -> dict[str, Any]:
    rows = []
    for case in cases:
        ragas_label = score_to_label(case.ragas_answer_relevancy)
        rows.append(
            {
                **asdict(case),
                "ragas_label": ragas_label,
                "label_agreement": ragas_label == case.human_label,
                "absolute_error": abs(
                    case.ragas_answer_relevancy - LABEL_VALUES[case.human_label]
                ),
            }
        )
    exact_agreement = sum(row["label_agreement"] for row in rows) / len(rows)
    return {
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "exact_label_agreement": exact_agreement,
            "mean_absolute_error": sum(row["absolute_error"] for row in rows)
            / len(rows),
            "acceptable": exact_agreement >= 0.80,
            "answer_relevancy_release_gate_enabled": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_calibration_report(load_calibration_cases(args.path))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
