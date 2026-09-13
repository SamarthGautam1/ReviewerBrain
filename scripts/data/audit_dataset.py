"""Dataset audit for ReviewerBrain raw reviewer datasets.

Read-only pass over the raw JSONL files (a-j metrics from the original
project audit). Prints the full report to stdout; redirect stdout to a
file to save it. Never writes any file itself.
"""
import json
import os
import secrets
import statistics
import sys
from collections import Counter

from reviewerbrain import paths

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Extensions treated as docs/config for the >20% flag and the
# "source code only" funnel stage.
DOCS_CONFIG_EXT = {
    ".md", ".rst", ".txt", ".adoc", ".tex", ".yaml", ".yml", ".json",
    ".toml", ".ini", ".cfg", ".conf", ".xml", ".html", ".htm", ".csv",
    ".tsv", ".svg", ".lock", ".sum", ".proto", ".pb.go",
}

rng = secrets.SystemRandom()


def ext_of(path):
    if not path:
        return "<no-path>"
    base = os.path.basename(path.strip())
    if "." not in base or (base.startswith(".") and base.count(".") == 1):
        return "<no-ext>"
    return "." + base.rsplit(".", 1)[1].lower()


def is_pure_suggestion(body):
    """Body consists solely of one ```suggestion fenced block."""
    b = body.strip()
    if not b.lower().startswith("```suggestion"):
        return False
    return b.endswith("```") and b.count("```") == 2


def classify(ext):
    return "docs/config" if ext in DOCS_CONFIG_EXT else "source/other"


def audit(name, filepath):
    n_prs = 0
    comments = []
    field_missing = Counter()
    pr_comment_counts = []
    created_min, created_max = None, None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pr = json.loads(line)
            n_prs += 1
            rc = pr.get("review_comments") or []
            pr_comment_counts.append(len(rc))
            for c in rc:
                c["_pr_number"] = pr.get("pr_number")
                c["_pr_title"] = pr.get("pr_title")
                comments.append(c)
                ca = c.get("created_at")
                if ca:
                    if created_min is None or ca < created_min:
                        created_min = ca
                    if created_max is None or ca > created_max:
                        created_max = ca
                for k in ("path", "diff_hunk", "body", "is_code_related",
                          "led_to_code_change", "in_reply_to_id",
                          "follow_up_patch"):
                    if c.get(k) is None:
                        field_missing[k] += 1

    n = len(comments)

    code_related = sum(1 for c in comments if c.get("is_code_related") is True)
    led_change = sum(1 for c in comments if c.get("led_to_code_change") is True)

    replies = sum(1 for c in comments if c.get("in_reply_to_id") is not None)
    fresh = n - replies

    ext_counter = Counter(ext_of(c.get("path")) for c in comments)
    docs_config = sum(v for k, v in ext_counter.items()
                      if classify(k) == "docs/config")
    docs_pct = 100.0 * docs_config / n if n else 0.0

    blens = [len(c.get("body") or "") for c in comments]
    blens_nz = [L for L in blens if L > 0]

    followups = sum(1 for c in comments
                    if (c.get("follow_up_patch") or "").strip())

    dup_key = Counter(
        (c.get("_pr_number"), c.get("path"), c.get("body")) for c in comments
    )
    dup_extra = sum(v - 1 for v in dup_key.values() if v > 1)
    dup_groups = sum(1 for v in dup_key.values() if v > 1)

    # Funnel of the cleaning filters, applied in the approved order.
    f_len = [c for c in comments if len(c.get("body") or "") >= 5]
    f_sug = [c for c in f_len if not is_pure_suggestion(c.get("body") or "")]
    f_fresh = [c for c in f_sug if c.get("in_reply_to_id") is None]
    f_src = [c for c in f_fresh
             if c.get("path") and classify(ext_of(c.get("path"))) == "source/other"]
    seen = set()
    f_dedup = []
    for c in f_src:
        k = (c.get("_pr_number"), c.get("path"), c.get("body"))
        if k not in seen:
            seen.add(k)
            f_dedup.append(c)

    tok = sum((len(c.get("diff_hunk") or "") + len(c.get("body") or "")) / 4.0
              for c in f_dedup)
    tok_all = sum((len(c.get("diff_hunk") or "") + len(c.get("body") or "")) / 4.0
                  for c in comments)
    tok_per_ex = tok / len(f_dedup) if f_dedup else 0

    pool = [c for c in comments
            if c.get("in_reply_to_id") is None
            and (c.get("diff_hunk") or "").strip()
            and (c.get("body") or "").strip()]
    examples = rng.sample(pool, 5) if len(pool) >= 5 else pool

    return {
        "name": name, "filepath": filepath, "n_prs": n_prs, "n": n,
        "field_missing": field_missing,
        "code_related": code_related, "led_change": led_change,
        "replies": replies, "fresh": fresh,
        "ext_counter": ext_counter, "docs_config": docs_config,
        "docs_pct": docs_pct,
        "blens_nz": blens_nz, "empty_bodies": blens.count(0),
        "followups": followups,
        "dup_extra": dup_extra, "dup_groups": dup_groups,
        "funnel": {"all": n, "len>=5": len(f_len),
                   "not-suggestion": len(f_sug), "fresh-only": len(f_fresh),
                   "source-only": len(f_src), "deduped": len(f_dedup)},
        "tok": tok, "tok_all": tok_all, "tok_per_ex": tok_per_ex,
        "examples": examples,
        "pr_comment_counts": pr_comment_counts,
        "created_min": created_min, "created_max": created_max,
    }


def fmt_stats(lens):
    if not lens:
        return "n/a"
    return (f"min={min(lens)}  max={max(lens)}  "
            f"median={statistics.median(lens):.0f}  "
            f"avg={sum(lens)/len(lens):.0f}")


def main():
    results = [audit(name, path) for name, path in paths.RAW_FILES.items()]
    out = ["# ReviewerBrain — Dataset Audit", ""]

    for r in results:
        out += [
            f"## Reviewer: {r['name']}",
            f"File: `{os.path.basename(r['filepath'])}`",
            "",
            f"- (a) PRs total: **{r['n_prs']}**",
            f"- (b) review_comments total (flattened): **{r['n']}**",
            f"- (c) is_code_related=True: **{r['code_related']}** "
            f"({100*r['code_related']/r['n']:.1f}%)",
            f"- (d) led_to_code_change=True: **{r['led_change']}** "
            f"({100*r['led_change']/r['n']:.1f}%)",
            f"- (e) replies (in_reply_to_id set): **{r['replies']}** vs "
            f"fresh top-level: **{r['fresh']}** "
            f"({100*r['replies']/r['n']:.1f}% replies)",
            f"- (f) docs/config files: **{r['docs_config']}** of {r['n']} "
            f"= **{r['docs_pct']:.1f}%** "
            f"{'>>> FLAG: >20% docs/config <<<' if r['docs_pct'] > 20 else '(<=20%, ok)'}",
            f"- (g) body length chars — {fmt_stats(r['blens_nz'])}; "
            f"empty bodies: {r['empty_bodies']}",
            f"- (h) non-empty follow_up_patch: **{r['followups']}** "
            f"({100*r['followups']/r['n']:.1f}%)",
            f"- duplicates on (pr_number, path, body): {r['dup_extra']} extra "
            f"rows in {r['dup_groups']} duplicate groups",
            f"- comment created_at range: {r['created_min']} → {r['created_max']}",
            f"- comments per PR: min={min(r['pr_comment_counts'])} "
            f"max={max(r['pr_comment_counts'])} "
            f"avg={sum(r['pr_comment_counts'])/len(r['pr_comment_counts']):.1f}",
            f"- null/missing fields: {dict(r['field_missing']) or 'none'}",
            "",
            "### Full extension breakdown",
            "",
            "| ext | count | % | class |",
            "|---|---|---|---|",
        ]
        for ext, cnt in r["ext_counter"].most_common():
            out.append(f"| {ext} | {cnt} | {100*cnt/r['n']:.1f}% | "
                       f"{classify(ext)} |")
        out += [
            "",
            "### Filtering funnel (cleaning filters, applied in order)",
            "",
            "| stage | kept | dropped |",
            "|---|---|---|",
        ]
        prev = r["funnel"]["all"]
        for stage in ("len>=5", "not-suggestion", "fresh-only", "source-only",
                      "deduped"):
            kept = r["funnel"][stage]
            out.append(f"| {stage} | {kept} | -{prev-kept} |")
            prev = kept
        out += [
            "",
            "### Token estimate",
            "",
            f"- After basic filters: **{r['funnel']['deduped']} examples**, "
            f"~**{r['tok']:,.0f} tokens** total "
            f"(~{r['tok_per_ex']:.0f} tokens/example avg)",
            f"- Unfiltered reference: ~{r['tok_all']:,.0f} tokens",
            "",
        ]

        out.append("### 5 random full examples (fresh, non-empty hunk+body)")
        out.append("")
        for i, c in enumerate(r["examples"], 1):
            out += [
                f"#### Example {i} — PR #{c.get('_pr_number')} "
                f"“{c.get('_pr_title')}”",
                f"- path: `{c.get('path')}`  line: {c.get('line')}  "
                f"is_code_related: {c.get('is_code_related')}  "
                f"led_to_code_change: {c.get('led_to_code_change')}",
                "",
                "**diff_hunk:**",
                "```diff",
                (c.get("diff_hunk") or "").rstrip(),
                "```",
                "",
                "**body (target output):**",
                "```",
                (c.get("body") or "").rstrip(),
                "```",
                "",
            ]
        out.append("---")
        out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
