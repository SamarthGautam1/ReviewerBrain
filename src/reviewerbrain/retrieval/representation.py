"""Shared retrieval-representation logic for ReviewerBrain RAG (v2).

Comment-first representation so the review comment is always inside the
embedding window:

    Reviewer comment:
    <review_comment>

    File:
    <file path>

    Relevant diff:
    <trimmed diff context>

Exact truncation rule (documented in docs/methodology/rag_reproduction.md):
  1. usable window = model.max_seq_length (256 for all-MiniLM-L6-v2),
     minus 2 for [CLS]/[SEP].
  2. diff token budget = usable - tokens("Reviewer comment:\n" + comment
     + "\n\nFile:\n" + path + "\n\nRelevant diff:\n") - 2 (safety).
     The comment is never cut: it is placed first and the budget is taken
     out of the diff. If the comment itself is too long to ever fit, the
     example is flagged comment_overflows_window (counted, never deleted).
  3. diff budget floors at MIN_DIFF_TOKENS (24). When the floor is used the
     total may exceed the window; the model then truncates the diff TAIL
     only - the comment (at the front) is still fully embedded.
  4. diff trimming: the hunk's @@ header is parsed to give each body line a
     new-file line number. The anchor line is the one whose number matches
     the comment's `line` field (nearest if unmatchable; if `line` is null
     the anchor is the midpoint of the +/- changed lines, else the middle of
     the hunk). A contiguous window is grown outward from the anchor
     (nearest line first) until the token budget is hit. No elision
     markers. A single line larger than the whole budget is hard-truncated
     at a proportional character count and flagged.
  5. Query representation (held-out eval / production) has no comment:
     "File:\n<path>\n\nRelevant diff:\n<trimmed diff>" with the same trim
     rule; the freed comment budget goes to the diff.

follow_up_patch is never used anywhere.
"""
import json
import re

from reviewerbrain import paths

MIN_DIFF_TOKENS = 24
SAFETY = 2
HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

DOC_SCAFFOLD = "Reviewer comment:\n"     # + comment
DOC_MIDDLE = "\n\nFile:\n"               # + path
DOC_DIFF = "\n\nRelevant diff:\n"        # + diff
QUERY_SCAFFOLD = "File:\n"               # + path
QUERY_DIFF = "\n\nRelevant diff:\n"      # + diff


def load_enriched():
    """Load cleaned records joined (read-only) with line + comment_id from
    the raw datasets. Join key (reviewer, pr_number, file, body), first
    occurrence, fresh comments only - mirrors the cleaning filters."""
    clean = []
    for name, path in paths.CLEAN_FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    clean.append(json.loads(l))

    mapping = {}
    for name, path in paths.RAW_FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pr = json.loads(line)
                for c in pr.get("review_comments") or []:
                    if c.get("in_reply_to_id") is not None:
                        continue
                    key = (name, pr.get("pr_number"), c.get("path"),
                           c.get("body"))
                    if key not in mapping:
                        mapping[key] = (c.get("line"), c.get("comment_id"))

    misses = 0
    for r in clean:
        k = (r["reviewer"], r["pr_number"], r["file"], r["review_comment"])
        line, cid = mapping.get(k, (None, None))
        if k not in mapping:
            misses += 1
        r["_line"] = line
        r["_comment_id"] = cid
    return clean, misses


def parse_hunk(hunk):
    """Return (body_lines, new_file_line_numbers, flags)."""
    lines = hunk.split("\n")
    header_idx = None
    n_headers = 0
    for i, l in enumerate(lines):
        if HUNK_HEADER_RE.match(l):
            n_headers += 1
            if header_idx is None:
                header_idx = i
                start_new = int(HUNK_HEADER_RE.match(l).group(1))
    flags = {"no_header": header_idx is None, "multi_header": n_headers > 1}
    if header_idx is None:
        body = lines
        new_nos = [None] * len(body)
        return body, new_nos, flags
    body = lines[header_idx + 1:]
    new_nos = []
    no = start_new
    for l in body:
        if l.startswith("+") or l.startswith(" "):
            new_nos.append(no)
            no += 1
        elif l.startswith("-"):
            new_nos.append(None)
        else:  # "\ No newline" or stray
            new_nos.append(None)
    return body, new_nos, flags


def anchor_index(body, new_nos, line_target, flags):
    changed = [i for i, l in enumerate(body)
               if l[:1] in ("+", "-") and not l.startswith("---")]
    if line_target is not None and new_nos and any(n is not None
                                                   for n in new_nos):
        cands = [i for i, n in enumerate(new_nos) if n is not None]
        best = min(cands, key=lambda i: abs(new_nos[i] - line_target))
        flags["anchor_source"] = "matched_line" if new_nos[best] == line_target \
            else "nearest_line"
        return best
    if changed:
        flags["anchor_source"] = "changed_midpoint"
        return changed[len(changed) // 2]
    flags["anchor_source"] = "hunk_midpoint"
    return len(body) // 2


def trim_diff(hunk, line_target, tok, budget, info):
    body, new_nos, hflags = parse_hunk(hunk)
    info.update(hflags)
    if not body:
        info.update({"anchor_source": "empty_hunk", "lines_kept": 0,
                     "lines_total": 0, "diff_tokens": 0})
        return ""
    a = anchor_index(body, new_nos, line_target, info)
    costs = [len(tok.encode(l + "\n", add_special_tokens=False))
             for l in body]
    lo = hi = a
    used = costs[a]
    if used > budget:
        keep_chars = max(1, int(len(body[a]) * budget / used))
        body = body[:a] + [body[a][:keep_chars]] + body[a + 1:]
        costs[a] = len(tok.encode(body[a] + "\n", add_special_tokens=False))
        used = costs[a]
        info["oversized_anchor_line"] = True
    while used < budget and (lo > 0 or hi < len(body) - 1):
        progressed = False
        if hi < len(body) - 1 and used + costs[hi + 1] <= budget:
            hi += 1
            used += costs[hi]
            progressed = True
        if not progressed and lo > 0 and used + costs[lo - 1] <= budget:
            lo -= 1
            used += costs[lo]
            progressed = True
        if not progressed and hi < len(body) - 1 and used + costs[hi + 1] <= budget:
            hi += 1
            used += costs[hi]
            progressed = True
        if not progressed:
            break
    info["lines_kept"] = hi - lo + 1
    info["lines_total"] = len(body)
    info["diff_tokens"] = used
    return "\n".join(body[lo:hi + 1])


def doc_text(r, tok, max_len):
    info = {}
    comment = r["review_comment"]
    path = r["file"] or ""
    c_tok = len(tok.encode(comment, add_special_tokens=False))
    p_tok = len(tok.encode(path, add_special_tokens=False))
    s_tok = len(tok.encode(DOC_SCAFFOLD + DOC_MIDDLE + DOC_DIFF,
                           add_special_tokens=False))
    budget = max_len - 2 - SAFETY - c_tok - p_tok - s_tok
    info["comment_tokens"] = c_tok
    info["comment_overflows_window"] = (c_tok + p_tok + s_tok + 2 > max_len)
    if budget < MIN_DIFF_TOKENS:
        budget = MIN_DIFF_TOKENS
        info["budget_floored"] = True
    diff = trim_diff(r["diff_hunk"], r.get("_line"), tok, budget, info)
    text = (DOC_SCAFFOLD + comment + DOC_MIDDLE + path + DOC_DIFF + diff)
    info["total_tokens"] = len(tok.encode(text, add_special_tokens=True))
    return text, info


def query_text(r, tok, max_len):
    info = {}
    path = r["file"] or ""
    p_tok = len(tok.encode(path, add_special_tokens=False))
    s_tok = len(tok.encode(QUERY_SCAFFOLD + QUERY_DIFF,
                           add_special_tokens=False))
    budget = max_len - 2 - SAFETY - p_tok - s_tok
    diff = trim_diff(r["diff_hunk"], r.get("_line"), tok, budget, info)
    text = QUERY_SCAFFOLD + path + QUERY_DIFF + diff
    info["total_tokens"] = len(tok.encode(text, add_special_tokens=True))
    return text, info
