from evaluation.llm_judge import LLMJudge
from generation.llm_client import LLMCompletion


class FakeClient:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def complete(self, system, user, stream=False):
        self.calls += 1
        return LLMCompletion(self.text, "stop")


def test_judge_parses_and_caches_structured_verdicts(tmp_path):
    fake = FakeClient('{"verdicts":[{"subject_id":"c1","check":"claim_support","verdict":"supported","rationale":"passage says so"}]}')
    judge = LLMJudge(fake, judge_model="judge", model_under_test="answerer", cache_path=tmp_path / "judge.json")
    first = judge.judge(question_id="q1", question="q", answer="a", evidence=[], subjects=[])
    second = judge.judge(question_id="q1", question="q", answer="a", evidence=[], subjects=[])
    assert first.judge_status == "judged"
    assert second.cached is True
    assert fake.calls == 1


def test_judge_failure_is_unjudged_not_a_pass():
    result = LLMJudge(FakeClient("not json"), judge_model="judge", model_under_test="answerer").judge(question_id="q", question="q", answer="a", evidence=[], subjects=[])
    assert result.judge_status == "unjudged"
    assert result.verdicts == []


def test_judge_model_must_differ_from_answer_model():
    import pytest
    with pytest.raises(ValueError, match="differ"):
        LLMJudge(FakeClient("{}"), judge_model="same", model_under_test="same")