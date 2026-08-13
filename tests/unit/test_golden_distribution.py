from __future__ import annotations

from dataclasses import replace

import pytest

from evaluation.generation_golden import load_generation_golden
from evaluation.golden_distribution import validate_golden_distribution


def test_tracked_completeness_golden_has_balanced_distribution():
    questions = load_generation_golden("evaluation/fixtures/completeness_golden.json")

    assert validate_golden_distribution(questions) == {
        "direct_fact": 0.3,
        "mechanism_explanation": 0.4,
        "multi_part_synthesis": 0.3,
    }


def test_distribution_validator_rejects_drift_beyond_five_percent():
    questions = load_generation_golden("evaluation/fixtures/completeness_golden.json")
    drifted = [
        replace(question, benchmark_category="direct_fact") for question in questions
    ]

    with pytest.raises(ValueError, match="outside tolerance"):
        validate_golden_distribution(drifted, tolerance=0.05)
