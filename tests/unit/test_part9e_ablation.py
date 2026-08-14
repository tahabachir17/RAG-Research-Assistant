from evaluation.run_part9e_ablation import best_gold_rank, display_rank


def test_best_gold_rank_finds_various_positions():
    ranking = ["a", "b", "c", "d"]
    assert best_gold_rank(ranking, ["a"]) == 1
    assert best_gold_rank(ranking, ["c"]) == 3
    assert best_gold_rank(ranking, ["d"]) == 4


def test_best_gold_rank_returns_none_when_absent_or_candidates_empty():
    assert best_gold_rank(["a", "b"], ["missing"]) is None
    assert best_gold_rank([], ["a"]) is None


def test_best_gold_rank_uses_best_of_multiple_gold_chunks():
    assert best_gold_rank(["a", "gold-2", "b", "gold-1"], ["gold-1", "gold-2"]) == 2


def test_display_rank_uses_explicit_cutoff_vocabulary():
    assert display_rank(4) == "4"
    assert display_rank(20) == "20"
    assert display_rank(21) == "not in top-20"
    assert display_rank(None) == "not in top-20"
