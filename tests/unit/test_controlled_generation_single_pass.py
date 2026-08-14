from evaluation.run_controlled_generation_single_pass import parser


def test_single_pass_is_generation_only_with_controlled_defaults():
    args = parser().parse_args([])

    assert args.generator_provider == "groq"
    assert args.generator_model == "llama-3.3-70b-versatile"
    assert args.run_id == "controlled_generation_single_pass"
    assert not hasattr(args, "judge_model")


def test_single_pass_accepts_repeated_question_filters():
    args = parser().parse_args(
        ["--question-id", "controlled-qrnn-01", "--question-id", "controlled-nsm-02"]
    )

    assert args.question_id == ["controlled-qrnn-01", "controlled-nsm-02"]
