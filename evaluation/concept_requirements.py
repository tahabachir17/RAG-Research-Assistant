"""Validated, deterministic required-concept phrase specifications."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias


MAX_CONCEPT_ALIASES = 5
ConceptRequirement: TypeAlias = str | dict[str, Any]


def parse_required_concepts(value: Any) -> list[ConceptRequirement]:
    """Load string or structured concepts while preserving authored phrases."""

    if not isinstance(value, list):
        raise TypeError("required_concepts must be an array")
    result: list[ConceptRequirement] = []
    seen: set[str] = set()
    for raw in value:
        concept, aliases = concept_requirement_phrases(raw)
        key = _normalized_phrase(concept)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(
            concept if not aliases else {"concept": concept, "aliases": aliases}
        )
    return result


def concept_requirement_phrases(value: Any) -> tuple[str, list[str]]:
    """Return a primary concept and its bounded, ordered alias phrases."""

    if isinstance(value, str):
        concept = value.strip()
        if not concept:
            raise ValueError("required concept must not be empty")
        return concept, []
    if not isinstance(value, Mapping):
        raise TypeError("required concept must be a string or object")
    concept = str(value.get("concept", "")).strip()
    if not concept:
        raise ValueError("structured required concept needs a non-empty concept")
    raw_aliases = value.get("aliases", [])
    if not isinstance(raw_aliases, list):
        raise TypeError("required concept aliases must be an array")
    if len(raw_aliases) > MAX_CONCEPT_ALIASES:
        raise ValueError(
            f"required concept may define at most {MAX_CONCEPT_ALIASES} aliases"
        )
    aliases = list(
        dict.fromkeys(str(alias).strip() for alias in raw_aliases if str(alias).strip())
    )
    primary = _normalized_phrase(concept)
    aliases = [alias for alias in aliases if _normalized_phrase(alias) != primary]
    return concept, aliases


def _normalized_phrase(value: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in value)
        .casefold()
        .split()
    )
