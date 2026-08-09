from evaluation.generation_metrics import (
    claim_level_citation_coverage,
    citation_validity_rate,
    qualifying_item_precision,
    qualifying_item_recall,
    required_field_completeness,
    truncation_rate,
    unsupported_claim_rate,
)


def test_generation_metrics_are_pure_and_handle_denominators():
    assert qualifying_item_precision(["A", "wrong"], ["a", "b"]) == 0.5
    assert qualifying_item_recall(["A", "wrong"], ["a", "b"]) == 0.5
    assert citation_validity_rate([True, False, True]) == 2 / 3
    assert claim_level_citation_coverage([{"cited": True}, {"cited": False}]) == 0.5
    assert required_field_completeness(["dataset"], ["dataset", "benefit"]) == 0.5
    assert truncation_rate(["stop", "length", None]) == 1 / 3
    assert unsupported_claim_rate(["supported", "unsupported", "unjudged"]) == 0.5
    assert qualifying_item_precision([], []) is None
    assert qualifying_item_recall([], []) is None
    assert unsupported_claim_rate([]) is None
