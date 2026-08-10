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
        GeminiClient,
        LLMClient,
        LLMClientError,
        LLMCompletion,
        LLMStreamChunk,
        LMStudioClient,
        OllamaClient,
        OpenAIClient,
        build_llm_client,
    )
    from .provider_router import ProviderAttempt, ProviderRouter, build_zero_cost_router
    from .prompt_manager import PromptManager, PromptTemplate
    from .response_validator import ResponseValidator, ValidationResult, ValidatedGeneration, generate_with_validation
    from .response_formatter import GeneratedAnswer, format_response
    from .streaming_handler import stream_answer, stream_answer_events
    from .structured_answer import StructuredAnswerError, parse_and_render_structured_answer, parse_and_render_structured_narrative, render_structured_answer, structured_answer_instruction, structured_narrative_instruction
except ImportError:
    from citation_handler import (
        CitationValidationResult,
        build_source_list,
        validate_citations,
    )
    from context_assembler import AssembledContext, CitationSource, ContextAssembler
    from llm_client import (
        ClaudeClient,
        GeminiClient,
        LLMClient,
        LLMClientError,
        LLMCompletion,
        LLMStreamChunk,
        LMStudioClient,
        OllamaClient,
        OpenAIClient,
        build_llm_client,
    )
    from provider_router import ProviderAttempt, ProviderRouter, build_zero_cost_router
    from prompt_manager import PromptManager, PromptTemplate
    from response_validator import ResponseValidator, ValidationResult, ValidatedGeneration, generate_with_validation
    from response_formatter import GeneratedAnswer, format_response
    from streaming_handler import stream_answer, stream_answer_events
    from structured_answer import StructuredAnswerError, parse_and_render_structured_answer, parse_and_render_structured_narrative, render_structured_answer, structured_answer_instruction, structured_narrative_instruction
__all__ = [
    "AssembledContext",
    "CitationSource",
    "CitationValidationResult",
    "ClaudeClient",
    "ContextAssembler",
    "GeneratedAnswer",
    "GeminiClient",
    "LLMClient",
    "LLMClientError",
    "LLMCompletion",
    "LLMStreamChunk",
    "LMStudioClient",
    "OllamaClient",
    "OpenAIClient",
    "PromptManager",
    "PromptTemplate",
    "ProviderAttempt",
    "ProviderRouter",
    "ResponseValidator",
    "StructuredAnswerError",
    "ValidationResult",
    "ValidatedGeneration",
    "build_llm_client",
    "build_zero_cost_router",
    "build_source_list",
    "format_response",
    "generate_with_validation",
    "stream_answer",
    "stream_answer_events",
    "parse_and_render_structured_answer",
    "parse_and_render_structured_narrative",
    "render_structured_answer",
    "structured_answer_instruction",
    "structured_narrative_instruction",
    "validate_citations",
]
