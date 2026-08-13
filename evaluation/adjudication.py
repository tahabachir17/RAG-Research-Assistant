"""Optional second-judge adjudication for disputed correctness scores."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AdjudicationConfig:
    enabled: bool = False
    correctness_threshold: float = 0.75
    high_concept_recall: float = 0.80
    score_disagreement_gap: float = 0.40
    judge_disagreement_gap: float = 0.20


def adjudication_reasons(
    *,
    primary_correctness: float | None,
    concept_recall: float | None,
    answer_relevancy: float | None,
    config: AdjudicationConfig,
) -> list[str]:
    """Return only the configured dispute conditions that apply."""

    if not config.enabled or primary_correctness is None:
        return []
    reasons: list[str] = []
    if primary_correctness < config.correctness_threshold:
        reasons.append("correctness_below_threshold")
    if (
        concept_recall is not None
        and concept_recall >= config.high_concept_recall
        and primary_correctness < config.correctness_threshold
    ):
        reasons.append("high_concept_recall_low_correctness")
    if (
        answer_relevancy is not None
        and abs(primary_correctness - answer_relevancy)
        >= config.score_disagreement_gap
    ):
        reasons.append("correctness_relevancy_disagreement")
    return reasons


def adjudicate_disputed_case(
    *,
    primary_correctness: float | None,
    concept_recall: float | None,
    answer_relevancy: float | None,
    config: AdjudicationConfig,
    secondary_judge: Callable[[], float],
    generator_model: str,
    primary_judge_model: str,
    secondary_judge_model: str,
) -> dict[str, Any]:
    """Run the secondary judge only for disputes and retain both scores."""

    reasons = adjudication_reasons(
        primary_correctness=primary_correctness,
        concept_recall=concept_recall,
        answer_relevancy=answer_relevancy,
        config=config,
    )
    result: dict[str, Any] = {
        "triggered": bool(reasons),
        "trigger_reasons": reasons,
        "primary_correctness": primary_correctness,
        "secondary_correctness": None,
        "primary_judge_model": primary_judge_model,
        "secondary_judge_model": secondary_judge_model,
        "disagreement": False,
    }
    if not reasons:
        return result
    if secondary_judge_model == generator_model:
        raise ValueError("secondary judge model must remain separate from generator model")
    if secondary_judge_model == primary_judge_model:
        raise ValueError("secondary judge model must differ from primary judge model")
    secondary = float(secondary_judge())
    result["secondary_correctness"] = secondary
    result["disagreement"] = (
        primary_correctness is not None
        and abs(primary_correctness - secondary) >= config.judge_disagreement_gap
    )
    return result
