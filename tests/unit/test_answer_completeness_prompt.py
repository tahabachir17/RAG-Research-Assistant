from __future__ import annotations

from generation.prompt_manager import PromptManager


def test_qa_prompt_contains_explicit_completeness_contract():
    system, _ = PromptManager().render(
        "qa_prompt",
        context=(
            "[1] QRNNs use convolutional computation in parallel and recurrent "
            "pooling for long-distance context and sequence order."
        ),
        question=(
            "How do QRNNs combine the useful properties of convolutional and "
            "recurrent sequence models?"
        ),
    )

    assert "COMPLETENESS CONTRACT:" in system
    assert "identify every distinct item" in system
    assert "every mechanism, reason, limitation, result, or future improvement" in system
    assert "never replace a concrete mechanism with a vague general description" in system
