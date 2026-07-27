"""Consistent query cleaning, BM25 tokenization, and deterministic expansion."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping, Sequence

from processing.bm25_indexer import BM25Indexer, DEFAULT_PREPROCESSING_CONFIG

logger = logging.getLogger(__name__)

DEFAULT_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "rag": ("retrieval augmented generation",),
    "llm": ("large language model",),
    "bm25": ("best matching 25", "lexical retrieval"),
    "evaluation": ("assessment", "benchmarking"),
}


@dataclass(frozen=True, slots=True)
class ProcessedQuery:
    original_query: str
    cleaned_query: str
    dense_query: str
    sparse_tokens: list[str]
    expanded_query: str | None = None


class QueryProcessor:
    """Prepare one query consistently for dense and sparse retrieval."""

    def __init__(
        self,
        *,
        enable_expansion: bool = False,
        expansion_dictionary: Mapping[str, str | Sequence[str]] | None = None,
        preprocessing_config: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(enable_expansion, bool):
            raise TypeError("enable_expansion must be a boolean")
        self.enable_expansion = enable_expansion
        self.preprocessing_config = _validate_preprocessing_config(preprocessing_config)
        self.expansion_dictionary = _validate_expansions(
            expansion_dictionary
            if expansion_dictionary is not None
            else DEFAULT_EXPANSIONS
        )

    def clean(self, query: str) -> str:
        """Normalize whitespace/control characters without damaging technical terms."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        normalized = unicodedata.normalize("NFC", query)
        normalized = "".join(
            " " if character in "\r\n\t" else character
            for character in normalized
            if character in "\r\n\t"
            or not unicodedata.category(character).startswith("C")
        )
        cleaned = re.sub(r"\s+", " ", normalized, flags=re.UNICODE).strip()
        if not cleaned:
            raise ValueError("query must not be empty or whitespace-only")
        return cleaned

    def tokenize_for_sparse(self, query: str) -> list[str]:
        """Use the exact preprocessing contract exported by ``BM25Indexer``."""

        cleaned = self.clean(query)
        return BM25Indexer.tokenize(cleaned, self.preprocessing_config)

    def expand(self, query: str) -> str:
        """Append bounded acronym/synonym expansions while retaining the query."""

        cleaned = self.clean(query)
        additions: list[str] = []
        folded = cleaned.casefold()
        for term, expansions in self.expansion_dictionary.items():
            if re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", folded):
                additions.extend(
                    expansion
                    for expansion in expansions
                    if expansion.casefold() not in folded
                )
        return " ".join([cleaned, *dict.fromkeys(additions)])

    def process(self, query: str) -> ProcessedQuery:
        cleaned = self.clean(query)
        expanded = self.expand(cleaned) if self.enable_expansion else None
        retrieval_query = expanded or cleaned
        processed = ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned,
            dense_query=retrieval_query,
            sparse_tokens=self.tokenize_for_sparse(retrieval_query),
            expanded_query=expanded,
        )
        logger.debug(
            "Processed query into %d sparse tokens", len(processed.sparse_tokens)
        )
        return processed


def _validate_preprocessing_config(
    value: Mapping[str, object] | None,
) -> dict[str, object]:
    if value is not None and not isinstance(value, Mapping):
        raise TypeError("preprocessing_config must be a mapping")
    config = dict(DEFAULT_PREPROCESSING_CONFIG)
    config.update(value or {})
    tokenizer = config.get("tokenizer")
    if tokenizer not in {"word_v1", "technical_terms_v2"}:
        raise ValueError(f"Unsupported BM25 tokenizer: {tokenizer!r}")
    for key in ("lowercase", "stop_words_removed"):
        if not isinstance(config.get(key), bool):
            raise ValueError(f"preprocessing_config[{key!r}] must be a boolean")
    if config["stop_words_removed"]:
        raise ValueError(
            "stop-word removal is not supported by the current BM25 indexer"
        )
    return config


def _validate_expansions(
    values: Mapping[str, str | Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(values, Mapping):
        raise TypeError("expansion_dictionary must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for key, raw_expansions in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("expansion dictionary keys must be non-empty strings")
        expansions = (
            (raw_expansions,) if isinstance(raw_expansions, str) else raw_expansions
        )
        if not isinstance(expansions, Sequence) or isinstance(
            expansions, (bytes, bytearray)
        ):
            raise TypeError(f"expansions for {key!r} must be a string or sequence")
        cleaned = tuple(
            expansion.strip()
            for expansion in expansions
            if isinstance(expansion, str) and expansion.strip()
        )
        if len(cleaned) != len(expansions) or not cleaned:
            raise ValueError(f"expansions for {key!r} must be non-empty strings")
        result[key.strip()] = cleaned
    return result
