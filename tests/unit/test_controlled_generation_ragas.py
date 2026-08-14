from evaluation.run_controlled_generation_ragas import parser


def test_controlled_generation_ragas_uses_requested_judge_fallback_defaults():
    args = parser().parse_args([])

    assert args.judge_provider == "gemini"
    assert args.judge_model == "gemini-3.5-flash-lite"
    assert args.fallback_judge_provider == "groq"
    assert args.fallback_judge_model == "llama-3.1-8b-instant"
    assert args.generator_model == "llama-3.3-70b-versatile"
