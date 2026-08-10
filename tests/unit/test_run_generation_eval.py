from __future__ import annotations

from config import Settings
from evaluation.run_generation_eval import (
    _evaluation_exit_code,
    _resolve_evaluation_target,
    _resolve_generation_target,
    _validate_credentials,
)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        LLM_PROVIDER="groq",
        LLM_MODEL="answer-70b",
        GROQ_MODEL="generic-groq",
        GEMINI_MODEL="gemini-judge",
        LMSTUDIO_MODEL="qwen-local",
        JUDGE_PROVIDER="groq",
        JUDGE_MODEL="judge-8b",
    )


def test_explicit_same_provider_preserves_configured_models():
    settings = _settings()
    assert _resolve_generation_target(settings, "groq", None) == (
        "groq",
        "answer-70b",
    )
    assert _resolve_evaluation_target(settings, "groq", None) == (
        "groq",
        "judge-8b",
    )


def test_provider_switch_selects_provider_specific_model():
    settings = _settings()
    assert _resolve_generation_target(settings, "gemini", None) == (
        "gemini",
        "gemini-judge",
    )
    assert _resolve_evaluation_target(settings, "gemini", None) == (
        "gemini",
        "gemini-judge",
    )
    assert _resolve_evaluation_target(settings, "qwen", None) == (
        "qwen",
        "qwen-local",
    )


def test_explicit_models_always_win():
    settings = _settings()
    assert _resolve_generation_target(settings, "gemini", "chosen-answer") == (
        "gemini",
        "chosen-answer",
    )
    assert _resolve_evaluation_target(settings, "qwen", "chosen-judge") == (
        "qwen",
        "chosen-judge",
    )


def test_credentials_are_checked_for_generation_and_evaluation_matrix():
    settings = _settings()
    settings.GROQ_API_KEY = "groq-key"
    settings.JUDGE_PROVIDER = "gemini"
    settings.OPENAI_API_KEY = "existing-gemini-key"
    _validate_credentials(settings, judge_enabled=True, ragas_enabled=True)

    settings.OPENAI_API_KEY = None
    settings.GEMINI_API_KEY = None
    try:
        _validate_credentials(settings, judge_enabled=True, ragas_enabled=True)
    except ValueError as exc:
        assert "GEMINI_API_KEY" in str(exc)
    else:
        raise AssertionError("missing Gemini credentials must fail before generation")


def test_requested_evaluation_layers_must_complete():
    complete = {
        "aggregate": {"judge_coverage": 1.0},
        "ragas": {"status": "completed"},
    }
    assert _evaluation_exit_code(
        complete, judge_required=True, ragas_required=True
    ) == 0

    partial_ragas = {
        "aggregate": {"judge_coverage": 1.0},
        "ragas": {"status": "partial", "reason": "one metric unavailable"},
    }
    assert _evaluation_exit_code(
        partial_ragas, judge_required=True, ragas_required=True
    ) == 2

    missing_judge = {
        "aggregate": {"judge_coverage": 0.5},
        "ragas": {"status": "completed"},
    }
    assert _evaluation_exit_code(
        missing_judge, judge_required=True, ragas_required=True
    ) == 2
