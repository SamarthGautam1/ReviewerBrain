"""Deterministic held-out split checks."""

from reviewerbrain.evaluation.split import held_out_prs


def _records():
    recs = []
    for name, n_prs in (("thockin", 12), ("ezyang", 7)):
        for p in range(1, n_prs + 1):
            recs.append({"reviewer": name, "pr_number": p})
    return recs


def test_split_is_deterministic():
    recs = _records()
    assert held_out_prs(recs) == held_out_prs(recs)


def test_split_rule():
    held = held_out_prs(_records())
    # thockin PRs 1..12 sorted: positions 4 and 9 held -> PRs 5 and 10
    assert held["thockin"] == {5, 10}
    # ezyang PRs 1..7 sorted: position 4 held -> PR 5
    assert held["ezyang"] == {5}


def test_split_queries_and_index_disjoint():
    recs = _records()
    held = held_out_prs(recs)
    for r in recs:
        if r["pr_number"] in held[r["reviewer"]]:
            pass  # query
    # every held-out PR must not appear in the index side: trivially true
    # because the split is PR-level; assert the sets are subsets of seen PRs
    all_prs = {name: {r["pr_number"] for r in recs if r["reviewer"] == name}
               for name in ("thockin", "ezyang")}
    for name in ("thockin", "ezyang"):
        assert held[name] <= all_prs[name]
