"""Validate the required difficulty mix for completeness benchmarks."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

try:
    from .generation_golden import GenerationGoldenQuestion, load_generation_golden
except ImportError:
    from generation_golden import GenerationGoldenQuestion, load_generation_golden


EXPECTED_DISTRIBUTION = {
    "direct_fact": 0.30,
    "mechanism_explanation": 0.40,
    "multi_part_synthesis": 0.30,
}


def validate_golden_distribution(
    questions: Sequence[GenerationGoldenQuestion], *, tolerance: float = 0.05
) -> dict[str, float]:
    """Return observed proportions or raise when any band drifts too far."""

    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("tolerance must be between 0 and 1")
    if not questions:
        raise ValueError("golden set must contain questions")
    counts = Counter(question.benchmark_category for question in questions)
    unknown = sorted(set(counts) - set(EXPECTED_DISTRIBUTION))
    if unknown:
        raise ValueError(f"unknown benchmark categories: {', '.join(unknown)}")
    observed = {
        category: counts[category] / len(questions)
        for category in EXPECTED_DISTRIBUTION
    }
    drifted = [
        category
        for category, target in EXPECTED_DISTRIBUTION.items()
        if abs(observed[category] - target) > tolerance + 1e-12
    ]
    if drifted:
        detail = ", ".join(
            f"{category}={observed[category]:.1%} (target {EXPECTED_DISTRIBUTION[category]:.1%})"
            for category in drifted
        )
        raise ValueError(f"golden question distribution is outside tolerance: {detail}")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    validate_golden_distribution(
        load_generation_golden(args.path), tolerance=args.tolerance
    )
    print(f"Validated balanced golden set: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
