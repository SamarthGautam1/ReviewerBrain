"""Held-out query selection and clean-record join for review generation.

Deterministic, no RNG: the pilot sample takes evenly-spaced entries over
the per-reviewer held-out list (same "every n-th" scheme the held-out
retrieval report used), so baseline and RAG runs can be pointed at the
exact same queries.
"""
import json

from reviewerbrain import paths

QUERY_DIFF_MAX_CHARS = 6000   # LLM-facing diff cap (representation is untouched)
DESC_MAX_CHARS = 1000


def load_heldout():
    """All held-out query records, in artifact (file) order."""
    out = []
    with open(paths.HELDOUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def select_queries(heldout, reviewer, n):
    """Evenly-spaced deterministic sample of n held-out queries for one
    reviewer, in artifact order. Returns held-out records."""
    rows = [r for r in heldout if r["reviewer"] == reviewer]
    if n >= len(rows):
        return list(rows)
    picks = sorted({i * len(rows) // n for i in range(n)})
    return [rows[i] for i in picks]


def load_clean_for_heldout(held):
    """Join a held-out query record back to its full cleaned record
    (diff_hunk, pr_title, pr_description) via query_id = '<reviewer>:<row>'.
    Row = position in that reviewer's clean file, the same per-reviewer
    ordering build_indexes.py used. Verifies the join before returning;
    a mismatch raises (never silently mis-joins)."""
    name, row = held["query_id"].split(":")
    row = int(row)
    with open(paths.CLEAN_FILES[name], "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == row:
                rec = json.loads(line)
                break
        else:
            raise ValueError(f"query_id {held['query_id']}: row {row} out of "
                             f"range for {paths.CLEAN_FILES[name].name}")
    if rec["reviewer"] != held["reviewer"]:
        raise ValueError(f"query_id {held['query_id']}: reviewer mismatch "
                         f"({rec['reviewer']} != {held['reviewer']})")
    if rec["pr_number"] != held["query_pr_number"]:
        raise ValueError(f"query_id {held['query_id']}: pr_number mismatch "
                         f"({rec['pr_number']} != {held['query_pr_number']})")
    if rec["review_comment"] != held["query_review_comment"]:
        raise ValueError(f"query_id {held['query_id']}: ground-truth comment "
                         "mismatch")
    if held["file"] and rec["file"] != held["file"]:
        raise ValueError(f"query_id {held['query_id']}: file mismatch")
    return rec


def pr_context(rec):
    """Optional PR title/description block for the prompt ('' if absent)."""
    parts = []
    if rec.get("pr_title"):
        parts.append(f"Pull request title: {rec['pr_title'].strip()}")
    desc = (rec.get("pr_description") or "").strip()
    if desc:
        if len(desc) > DESC_MAX_CHARS:
            desc = desc[:DESC_MAX_CHARS] + "… [description truncated]"
        parts.append(f"Pull request description: {desc}")
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def cap_diff(diff, max_chars=QUERY_DIFF_MAX_CHARS):
    """Character-cap a diff hunk for the LLM prompt (representation and
    indexes are untouched; this only bounds the prompt size)."""
    diff = diff or ""
    if len(diff) > max_chars:
        return diff[:max_chars] + "\n[diff truncated]"
    return diff
