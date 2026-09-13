"""Deterministic held-out split rule for ReviewerBrain evaluation.

Rule (frozen): per reviewer, take the sorted unique PR numbers and hold out
every 5th one (position % 5 == 4). No RNG is involved, so the split is
reproducible on any machine. Because the split is PR-level, neither the
query example nor any sibling comment from the same PR can appear in the
retrieval index.
"""

PERIOD = 5
HELD_POSITION = 4
REVIEWERS = ("thockin", "ezyang")


def held_out_prs(records):
    """Return {reviewer: set(held-out pr_number)} for the given records."""
    held = {}
    for name in REVIEWERS:
        prs = sorted({r["pr_number"] for r in records
                      if r["reviewer"] == name})
        held[name] = {p for i, p in enumerate(prs)
                      if i % PERIOD == HELD_POSITION}
    return held
