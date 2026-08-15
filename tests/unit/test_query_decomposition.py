from __future__ import annotations

from generation.entities import extract_named_papers, is_multi_paper_question
from generation.query_decomposition import retrieve_per_entity
from retrieval.models import RetrievalResult


def _result(chunk_id: str, title: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"Evidence from {title}",
        score=score,
        source="test",
        paper_id=title.casefold().replace(" ", "-"),
        title=title,
        section="method",
    )


def test_extract_named_papers_handles_quote_styles_and_deduplicates():
    question = "Compare 'Paper A', “Paper B”, and 'paper a'."

    assert extract_named_papers(question) == ["Paper A", "Paper B"]
    assert is_multi_paper_question(question) is True


def test_unquoted_known_titles_are_detected_in_mention_order():
    known = {
        "Quasi-Recurrent Neural Networks",
        "Neural Enquirer: Learning to Query Tables with Natural Language",
        "Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision",
    }
    question = (
        "Compare Quasi-Recurrent Neural Networks, Neural Enquirer: Learning to "
        "Query Tables with Natural Language, and Neural Symbolic Machines: "
        "Learning Semantic Parsers on Freebase with Weak Supervision."
    )

    assert extract_named_papers(question, known) == [
        "Quasi-Recurrent Neural Networks",
        "Neural Enquirer: Learning to Query Tables with Natural Language",
        "Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision",
    ]
    assert is_multi_paper_question(question, known)


def test_known_title_match_prefers_long_title_over_embedded_short_title():
    question = "Compare Neural Symbolic Machines Extended and Paper B."

    assert extract_named_papers(
        question, {"Neural Symbolic Machines", "Neural Symbolic Machines Extended", "Paper B"}
    ) == ["Neural Symbolic Machines Extended", "Paper B"]


def test_entity_retrieval_filters_wrong_papers_and_balances_named_evidence():
    calls: list[tuple[str, int, int]] = []

    class Retriever:
        def search(self, query, top_k=None, *, candidate_top_k=None):
            calls.append((query, top_k, candidate_top_k))
            if '"Paper A"' in query and '"Paper B"' not in query:
                return [_result("wrong-a", "Paper B", 9), _result("a1", "Paper A", 8)]
            if '"Paper B"' in query and '"Paper A"' not in query:
                return [_result("wrong-b", "Paper A", 9), _result("b1", "Paper B", 8)]
            return [_result("general", "Other Paper", 10)]

    ranked, reports = retrieve_per_entity(
        "Compare 'Paper A' and 'Paper B', including their limitations.",
        "unused.pkl",
        retriever=Retriever(),
        per_entity_top_k=2,
        candidate_k=30,
    )

    assert [result.chunk_id for result in ranked[:2]] == ["a1", "b1"]
    assert [report.hit for report in reports] == [True, True]
    assert all(top_k == 30 and candidate_top_k == 30 for _, top_k, candidate_top_k in calls)


def test_entity_retrieval_reports_a_named_paper_without_matching_evidence():
    class Retriever:
        def search(self, query, **kwargs):
            if '"Paper A"' in query and '"Paper B"' not in query:
                return [_result("a1", "Paper A")]
            return [_result("other", "Other Paper")]

    ranked, reports = retrieve_per_entity(
        "Compare 'Paper A' and 'Paper B'.",
        "unused.pkl",
        retriever=Retriever(),
        include_general_query=False,
    )

    assert [result.chunk_id for result in ranked] == ["a1"]
    assert reports[0].hit is True
    assert reports[1].hit is False


def test_unquoted_entity_queries_remove_other_known_titles():
    known = {"Paper A", "Paper B"}
    calls: list[str] = []

    class Retriever:
        def search(self, query, **kwargs):
            calls.append(query)
            title = "Paper A" if query.startswith('"Paper A"') else "Paper B"
            return [_result(title[-1].casefold(), title)]

    retrieve_per_entity(
        "Compare Paper A and Paper B on training.",
        "unused.pkl",
        retriever=Retriever(),
        known_titles=known,
        include_general_query=False,
    )

    assert calls[0].startswith('"Paper A"') and "Paper B" not in calls[0]
    assert calls[1].startswith('"Paper B"') and "Paper A" not in calls[1]
