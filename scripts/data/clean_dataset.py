"""Flatten PR wrappers and clean review comments into training examples.

Applies the user-approved cleaning filters, in decision order:
  1. drop replies (in_reply_to_id not null)
  2. drop body < 5 chars
  3. drop pure ```suggestion``` blocks
  4. keep source-code files only
  5. dedup on (pr_number, path, body), keep first

Writes NO files itself. Emits one JSONL section per reviewer to stdout
(plus a ===REPORT=== section); pipe through scripts/split_sections.sh or
redirect manually to produce data/processed/thockin_clean.jsonl and
data/processed/ezyang_clean.jsonl, then concatenate both for
combined_clean.jsonl. After cleaning, run
scripts/data/apply_clean_amendments.py (two approved amendments).
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from reviewerbrain import paths

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = {"thockin": "kubernetes/kubernetes", "ezyang": "pytorch/pytorch"}

# Decision #4: clearly non-source documentation/configuration extensions.
# Everything NOT in this set is kept, including build/codegen files
# (.sh, .cmake, .expect, .proto, .mod, .sum, .lock, .patch, .bak).
NON_SOURCE_EXT = {
    ".md", ".rst", ".txt", ".adoc", ".tex",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".html", ".htm", ".csv", ".tsv", ".svg",
}
# Extensionless paths that are documentation rather than build files.
NON_SOURCE_BASENAMES = {
    "readme", "license", "notice", "authors", "contributors", "changelog",
    "code_of_conduct", "code-of-conduct", "security_contacts",
    "maintainers", "governance", "owners", "owner", "translations",
}

FIELDS = ["reviewer", "repo", "pr_number", "pr_title", "pr_description",
          "file", "diff_hunk", "review_comment", "is_code_related",
          "led_to_code_change", "follow_up_patch"]


def ext_of(path):
    base = os.path.basename(path.strip())
    if "." not in base:
        return "<no-ext>"
    return "." + base.rsplit(".", 1)[1].lower()


def is_source(path):
    if not path or not path.strip():
        return False
    base = os.path.basename(path.strip())
    stem = base.rsplit(".", 1)[0].lower() if "." in base else base.lower()
    if stem in NON_SOURCE_BASENAMES:
        return False
    return ext_of(path) not in NON_SOURCE_EXT


def is_pure_suggestion(body):
    """Body consists solely of one ```suggestion``` fenced block."""
    b = body.strip()
    if not b.lower().startswith("```suggestion"):
        return False
    return b.endswith("```") and b.count("```") == 2


def clean(name):
    n_prs = 0
    total = 0
    replies = 0
    short = 0
    pure_sug = 0
    non_source = 0
    dups = 0
    seen = set()
    records = []
    ext_all = Counter()
    no_ext_paths = set()
    prs_contributing = set()

    with open(paths.RAW_FILES[name], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pr = json.loads(line)
            n_prs += 1
            for c in pr.get("review_comments") or []:
                total += 1
                path = c.get("path")
                if path:
                    ext_all[ext_of(path)] += 1
                    if "." not in os.path.basename(path.strip()):
                        no_ext_paths.add(path)

                # decision 1: replies
                if c.get("in_reply_to_id") is not None:
                    replies += 1
                    continue
                # decision 2: minimum body length
                body = c.get("body") or ""
                if len(body) < 5:
                    short += 1
                    continue
                # decision 3: pure suggestion blocks
                if is_pure_suggestion(body):
                    pure_sug += 1
                    continue
                # decision 4: source files only
                if not is_source(path):
                    non_source += 1
                    continue
                # decision 5: dedup, keep first
                key = (pr.get("pr_number"), path, body)
                if key in seen:
                    dups += 1
                    continue
                seen.add(key)

                rec = {
                    "reviewer": name,
                    "repo": REPO[name],
                    "pr_number": pr.get("pr_number"),
                    "pr_title": pr.get("pr_title"),
                    "pr_description": pr.get("pr_description"),
                    "file": path,
                    "diff_hunk": c.get("diff_hunk"),
                    "review_comment": body,
                    "is_code_related": c.get("is_code_related"),
                    "led_to_code_change": c.get("led_to_code_change"),
                    "follow_up_patch": c.get("follow_up_patch"),
                }
                records.append(rec)
                prs_contributing.add(pr.get("pr_number"))

    return {
        "name": name, "n_prs": n_prs, "total": total, "replies": replies,
        "short": short, "pure_sug": pure_sug, "non_source": non_source,
        "dups": dups, "records": records, "ext_all": ext_all,
        "no_ext_paths": no_ext_paths,
        "prs_contributing": prs_contributing,
    }


def pct(x, d):
    return 100.0 * x / d if d else 0.0


def main():
    results = [clean("thockin"), clean("ezyang")]

    for r in results:
        print(f"==={r['name'].upper()}===")
        for rec in r["records"]:
            print(json.dumps(rec, ensure_ascii=False))

    out = ["# ReviewerBrain — Step 2 Cleaning Report", ""]

    out += [
        "## Extension classification applied (decision #4)",
        "",
        "Excluded as documentation/configuration: "
        + ", ".join(sorted(NON_SOURCE_EXT)),
        "",
        "Kept as source/build: `.sh`, `.cmake`, `.expect` plus other "
        "unlisted extensions (`.proto`, `.mod`, `.sum`, `.patch`, `.bak`, "
        "extensionless build files such as Makefile/Dockerfile). "
        "Extensionless docs (OWNERS, README, LICENSE, ...) excluded by "
        "basename.",
        "",
    ]
    for r in results:
        final = len(r["records"])
        led = sum(1 for x in r["records"] if x["led_to_code_change"] is True)
        fu = sum(1 for x in r["records"]
                 if (x["follow_up_patch"] or "").strip())
        related = sum(1 for x in r["records"] if x["is_code_related"] is True)
        out += [
            f"## Reviewer: {r['name']} (funnel, decision order)",
            "",
            f"- total PRs: {r['n_prs']} "
            f"(of which {len(r['prs_contributing'])} contribute ≥1 final example)",
            f"- total comments: {r['total']}",
            f"- replies removed: {r['replies']}",
            f"- body < 5 chars removed: {r['short']}",
            f"- pure suggestion blocks removed: {r['pure_sug']}",
            f"- non-source files removed: {r['non_source']}",
            f"- duplicates removed: {r['dups']}",
            f"- **final examples: {final}** "
            f"({pct(final, r['total']):.1f}% of original comments)",
            "",
            "Post-filter profile:",
            "",
            f"- led_to_code_change=True: {led} ({pct(led, final):.1f}%)",
            f"- non-empty follow_up_patch: {fu} ({pct(fu, final):.1f}%)",
            f"- is_code_related=True: {related} ({pct(related, final):.1f}%)",
            "",
        ]

    combined_total = sum(len(r["records"]) for r in results)
    grand_total = sum(r["total"] for r in results)
    out += [
        "## Combined",
        "",
        f"- thockin: {len(results[0]['records'])} examples "
        f"({pct(len(results[0]['records']), results[0]['total']):.1f}% retained)",
        f"- ezyang: {len(results[1]['records'])} examples "
        f"({pct(len(results[1]['records']), results[1]['total']):.1f}% retained)",
        f"- combined: **{combined_total} examples** "
        f"({pct(combined_total, grand_total):.1f}% of all "
        f"{grand_total} original comments)",
        "",
    ]

    print("===REPORT===")
    print("\n".join(out))


if __name__ == "__main__":
    main()
