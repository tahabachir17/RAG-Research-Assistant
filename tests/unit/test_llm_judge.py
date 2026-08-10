from evaluation.llm_judge import LLMJudge
from generation.llm_client import LLMCompletion


class FakeClient:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def complete(self, system, user, stream=False):
        self.calls += 1
        return LLMCompletion(self.text, "stop")


class FakeJsonClient(FakeClient):
    def __init__(self, responses):
        super().__init__("")
        self.responses = iter(responses)
        self.json_calls = 0

    def complete_json(self, system, user):
        self.json_calls += 1
        return LLMCompletion(next(self.responses), "stop")


def test_judge_parses_and_caches_structured_verdicts(tmp_path):
    fake = FakeClient('{"verdicts":[{"subject_id":"c1","check":"claim_support","verdict":"supported","rationale":"passage says so"}]}')
    judge = LLMJudge(fake, judge_model="judge", model_under_test="answerer", cache_path=tmp_path / "judge.json")
    subjects = [{"subject_id": "c1", "check": "claim_support"}]
    first = judge.judge(question_id="q1", question="q", answer="a", evidence=[], subjects=subjects)
    second = judge.judge(question_id="q1", question="q", answer="a", evidence=[], subjects=subjects)
    assert first.judge_status == "judged"
    assert second.cached is True
    assert fake.calls == 1


def test_judge_cache_changes_when_evidence_changes(tmp_path):
    fake = FakeClient('{"verdicts":[]}')
    judge = LLMJudge(fake, judge_model="judge", model_under_test="answerer", cache_path=tmp_path / "judge.json")
    judge.judge(question_id="q1", question="q", answer="a", evidence=[{"text": "first"}], subjects=[])
    judge.judge(question_id="q1", question="q", answer="a", evidence=[{"text": "changed"}], subjects=[])
    assert fake.calls == 2


def test_judge_uses_explicit_provider_label_for_local_qwen():
    judge = LLMJudge(
        FakeClient('{"verdicts":[]}'),
        judge_model="qwen-local",
        model_under_test="answerer",
        judge_provider="qwen",
    )
    assert judge.judge_provider == "qwen"


def test_judge_failure_is_unjudged_not_a_pass():
    result = LLMJudge(FakeClient("not json"), judge_model="judge", model_under_test="answerer").judge(question_id="q", question="q", answer="a", evidence=[], subjects=[])
    assert result.judge_status == "unjudged"
    assert result.verdicts == []


def test_judge_model_must_differ_from_answer_model():
    import pytest
    with pytest.raises(ValueError, match="differ"):
        LLMJudge(FakeClient("{}"), judge_model="same", model_under_test="same")


def test_judge_uses_native_json_mode_and_repairs_once():
    fake = FakeJsonClient(
        [
            "not json",
            '{"verdicts":[{"subject_id":"c1","check":"claim_support","verdict":"supported","rationale":"supported"}]}',
        ]
    )
    result = LLMJudge(
        fake,
        judge_model="judge",
        model_under_test="answerer",
    ).judge(
        question_id="q",
        question="question",
        answer="answer",
        evidence=[],
        subjects=[{"subject_id": "c1", "check": "claim_support"}],
    )
    assert result.judge_status == "judged"
    assert fake.json_calls == 2
    assert fake.calls == 0


def test_judge_rejects_incomplete_subject_coverage():
    fake = FakeJsonClient(
        [
            '{"verdicts":[{"subject_id":"c1","check":"claim_support","verdict":"supported","rationale":"ok"}]}',
            '{"verdicts":[{"subject_id":"c1","check":"claim_support","verdict":"supported","rationale":"ok"}]}',
        ]
    )
    result = LLMJudge(
        fake,
        judge_model="judge",
        model_under_test="answerer",
    ).judge(
        question_id="q",
        question="question",
        answer="answer",
        evidence=[],
        subjects=[
            {"subject_id": "c1", "check": "claim_support"},
            {"subject_id": "c2", "check": "claim_support"},
        ],
    )
    assert result.judge_status == "unjudged"
    assert "coverage mismatch" in (result.error or "")
