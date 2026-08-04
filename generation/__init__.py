"""Context assembly, prompting, LLM access, citations, and response formatting."""

from __future__ import annotations

try:
    from .citation_handler import (
        CitationValidationResult,
        build_source_list,
        validate_citations,
    )
    from .context_assembler import AssembledContext, CitationSource, ContextAssembler
    from .llm_client import (
        ClaudeClient,
        LLMClient,
        LLMClientError,
        OllamaClient,
        OpenAIClient,
        build_llm_client,
    )
    from .prompt_manager import PromptManager, PromptTemplate
    from .response_formatter import GeneratedAnswer, format_response
    from .streaming_handler import stream_answer, stream_answer_events
except ImportError:
    from citation_handler import (
        CitationValidationResult,
        build_source_list,
        validate_citations,
    )
    from context_assembler import AssembledContext, CitationSource, ContextAssembler
    from llm_client import (
        ClaudeClient,
        LLMClient,
        LLMClientError,
        OllamaClient,
        OpenAIClient,
        build_llm_client,
    )
    from prompt_manager import PromptManager, PromptTemplate
    from response_formatter import GeneratedAnswer, format_response
    from streaming_handler import stream_answer, stream_answer_events
__all__ = [
    "AssembledContext",
    "CitationSource",
    "CitationValidationResult",
    "ClaudeClient",
    "ContextAssembler",
    "GeneratedAnswer",
    "LLMClient",
    "LLMClientError",
    "OllamaClient",
    "OpenAIClient",
    "PromptManager",
    "PromptTemplate",
    "build_llm_client",
    "build_source_list",
    "format_response",
    "stream_answer",
    "stream_answer_events",
    "validate_citations",
]
