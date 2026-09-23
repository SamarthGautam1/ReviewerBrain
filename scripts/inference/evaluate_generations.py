"""Evaluate generated reviews against ground-truth reviewer comments.

Pairs the baseline and RAG runs produced by
scripts/inference/run_review_generation.py
(same held-out queries in both) and reports:

  - corpus BLEU-4 (uniform 4-gram weights + brevity penalty)
  - mean sentence-level BLEU-4 with add-1 smoothing (small-sample safe)
  - mean ROUGE-L F1 (LCS-based)
  - mean BERTScore F1 (only with --bertscore and the package installed;
    downloads a roberta-large checkpoint on first use)
  - generation counts, empty/error counts, output lengths

Plus a qualitative-examples markdown file (first N per reviewer):
current code / ground truth / baseline / RAG / retrieved examples.

These lexical/semantic similarity metrics DO NOT measure reviewer
replication — they are orientation numbers for the local baseline only
(the task's own caveat; interpretation lives in
evaluations/reports/local_llm_baseline.md).

Usage:
  python scripts/inference/evaluate_generations.py \
      --run-dir evaluations/inference/pilot_local [--bertscore]
"""
import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reviewerbrain.inference import outputs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOKEN_RE = re.compile(r"\w+")
DIFF_SHOW_CHARS = 1500
QUAL_PER_REVIEWER = 3


def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def ngram_counts(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def sentence_bleu4_smoothed(hyp, ref):
    """Sentence BLEU-4, add-1 smoothing (every n-gram statistic gets +1/+1),
    uniform weights, brevity penalty. Returns 0.0 for empty hypotheses."""
    h, r = tokens(hyp), tokens(ref)
    if not h or not r:
        return 0.0
    log_p = 0.0
    for n in range(1, 5):
        hc = ngram_counts(h, n)
        rc = ngram_counts(r, n)
        m = sum(min(c, rc[g]) for g, c in hc.items())
        t = max(sum(hc.values()), 0)
        p = (m + 1) / (t + 1)          # add-1 smoothing
        log_p += 0.25 * math.log(p)
    bp = min(1.0, math.exp(1.0 - len(r) / len(h)))
    return bp * math.exp(log_p)


def corpus_bleu4(hyps, refs):
    """Corpus BLEU-4: aggregated modified precision + brevity penalty.
    If any n-order has zero matches corpus-wide, falls back to add-1 for
    that order (noted in the returned dict)."""
    stats = {n: [0, 0] for n in range(1, 5)}
    hyp_len = ref_len = 0
    for hyp, ref in zip(hyps, refs):
        h, r = tokens(hyp), tokens(ref)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, 5):
            hc = ngram_counts(h, n)
            rc = ngram_counts(r, n)
            stats[n][0] += sum(min(c, rc[g]) for g, c in hc.items())
            stats[n][1] += max(sum(hc.values()), 0)
    smoothed = False
    log_p = 0.0
    for n in range(1, 5):
        m, t = stats[n]
        if m == 0:
            smoothed = True
            m, t = 1, t + 1 if t else 1
        log_p += 0.25 * math.log(m / t)
    bp = min(1.0, math.exp(1.0 - ref_len / hyp_len)) if hyp_len and ref_len \
        else 0.0
    return {"bleu4": bp * math.exp(log_p), "add1_fallback": smoothed,
            "hyp_tokens": hyp_len, "ref_tokens": ref_len}


def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[-1]


def rouge_l_f1(hyp, ref):
    h, r = tokens(hyp), tokens(ref)
    if not h or not r:
        return 0.0
    l = lcs_len(h, r)
    if l == 0:
        return 0.0
    p, rr = l / len(h), l / len(r)
    return 2 * p * rr / (p + rr)


def bert_score_pairs(hyps, refs):
    """Returns list of F1s or None when bert-score is unavailable."""
    try:
        import bert_score
    except ImportError:
        return None
    # package handles batching/model download; lang/en -> roberta-large
    scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=False)
    if not hyps:
        return []
    _, _, f1 = scorer.score(hyps, refs)
    return [float(x) for x in f1]


def load_pairs(run_dir, mode):
    """[(query_id, reviewer, hyp, ref, record)] for successfully generated
    records only; errors/empties are counted separately."""
    pairs, errors, empties = [], 0, 0
    for rec in outputs.read_generations(run_dir, mode):
        gen = rec.get("generated_comment")
        if rec.get("error") or gen is None:
            errors += 1
        elif not gen.strip():
            empties += 1
        else:
            pairs.append((rec["query_id"], rec["reviewer"], gen.strip(),
                          rec["ground_truth_comment"], rec))
    return pairs, errors, empties


def qualitative_md(run_dir, per_reviewer=QUAL_PER_REVIEWER):
    """Side-by-side examples for manual inspection (first N per reviewer,
    in run order)."""
    by_mode = {m: outputs.read_generations(run_dir, m)
               for m in ("baseline", "rag")}
    out = ["# Qualitative examples — local LLM pilot", "",
           "Fields: current code (capped), ground-truth review, baseline "
           "generation, RAG generation, retrieved historical examples.", ""]
    reviewers = sorted({r["reviewer"] for r in by_mode["rag"]})
    for name in reviewers:
        rag_recs = [r for r in by_mode["rag"] if r["reviewer"] == name]
        base_by_id = {r["query_id"]: r for r in by_mode["baseline"]}
        picks = rag_recs[:per_reviewer]
        out.append(f"## {name}")
        out.append("")
        for rec in picks:
            user_msg = rec["messages"][1]["content"]
            m = re.search(r"```diff\n(.*?)\n```", user_msg, re.DOTALL)
            code = m.group(1) if m else user_msg
            out.append(f"### {rec['query_id']} — PR#{rec['pr_number']} "
                       f"`{rec['file']}`")
            out.append("")
            out.append("**CURRENT CODE**")
            out.append("```diff")
            out.append(code[:DIFF_SHOW_CHARS] +
                       ("\n… [truncated]" if len(code) > DIFF_SHOW_CHARS
                        else ""))
            out.append("```")
            out.append("")
            out.append("**GROUND-TRUTH REVIEW**")
            out.append(f"> {rec['ground_truth_comment']}")
            out.append("")
            base = base_by_id.get(rec["query_id"], {})
            out.append("**BASELINE MODEL REVIEW**")
            out.append(f"> {base.get('generated_comment') or base.get('error')}")
            out.append("")
            out.append("**RAG MODEL REVIEW**")
            out.append(f"> {rec.get('generated_comment') or rec.get('error')}")
            out.append("")
            out.append(f"**RETRIEVED HISTORICAL REVIEWS** "
                       f"(n={len(rec.get('retrieved') or [])})")
            for ex in rec.get("retrieved") or []:
                c = ex["comment"].replace("\n", " ")
                c = c[:280] + ("…" if len(c) > 280 else "")
                out.append(f"- {ex['doc_id']} PR#{ex['pr_number']} "
                           f"`{ex['file']}` cos={ex['cos']:.3f} "
                           f"— {c}")
            out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True,
                    help="run directory containing baseline/ and rag/")
    ap.add_argument("--bertscore", action="store_true",
                    help="also compute BERTScore (needs the bert-score "
                         "package; downloads roberta-large on first use)")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)

    summary = {"run_dir": str(run_dir), "modes": {}}
    corpus_hyps = {"baseline": [], "rag": []}
    all_pairs = {}
    for mode in ("baseline", "rag"):
        pairs, errors, empties = load_pairs(run_dir, mode)
        all_pairs[mode] = pairs
        hyps = [p[2] for p in pairs]
        refs = [p[3] for p in pairs]
        corpus_hyps[mode] = hyps
        sb = [sentence_bleu4_smoothed(h, r) for h, r in zip(hyps, refs)]
        rl = [rouge_l_f1(h, r) for h, r in zip(hyps, refs)]
        entry = {
            "n_records": len(pairs) + errors + empties,
            "n_generated": len(pairs),
            "n_errors": errors,
            "n_empty": empties,
            "corpus_bleu4": corpus_bleu4(hyps, refs),
            "mean_sentence_bleu4_smoothed": sum(sb) / len(sb) if sb else 0.0,
            "mean_rouge_l_f1": sum(rl) / len(rl) if rl else 0.0,
            "avg_gen_words": (sum(len(tokens(h)) for h in hyps) / len(hyps)
                              if hyps else 0.0),
            "avg_gen_chars": (sum(len(h) for h in hyps) / len(hyps)
                              if hyps else 0.0),
            "avg_gt_words": (sum(len(tokens(r)) for r in refs) / len(refs)
                             if refs else 0.0),
        }
        summary["modes"][mode] = entry

    if args.bertscore:
        f1s = {}
        for mode in ("baseline", "rag"):
            hyps = corpus_hyps[mode]
            refs = [p[3] for p in all_pairs[mode]]
            got = bert_score_pairs(hyps, refs)
            f1s[mode] = (sum(got) / len(got)) if got else None
        summary["bertscore_f1_mean"] = f1s

    with outputs.open_run_file(run_dir, ".", "metrics.json") as f:
        import json
        json.dump(summary, f, indent=2)

    # markdown summary + cross-mode deltas
    b, r = summary["modes"]["baseline"], summary["modes"]["rag"]
    lines = [
        "# Local LLM pilot — automatic metrics", "",
        f"model: see manifest.json | pairs: baseline {b['n_generated']}, "
        f"rag {r['n_generated']} (errors: {b['n_errors']}/{r['n_errors']}, "
        f"empty: {b['n_empty']}/{r['n_empty']})", "",
        "| metric | baseline | rag |",
        "|---|---|---|",
        f"| corpus BLEU-4 | {b['corpus_bleu4']['bleu4']:.4f} | "
        f"{r['corpus_bleu4']['bleu4']:.4f} |",
        f"| mean sentence BLEU-4 (add-1) | "
        f"{b['mean_sentence_bleu4_smoothed']:.4f} | "
        f"{r['mean_sentence_bleu4_smoothed']:.4f} |",
        f"| mean ROUGE-L F1 | {b['mean_rouge_l_f1']:.4f} | "
        f"{r['mean_rouge_l_f1']:.4f} |",
    ]
    if "bertscore_f1_mean" in summary:
        lines.append(f"| mean BERTScore F1 | "
                     f"{summary['bertscore_f1_mean']['baseline']} | "
                     f"{summary['bertscore_f1_mean']['rag']} |")
    lines += [
        f"| avg generated words | {b['avg_gen_words']:.1f} | "
        f"{r['avg_gen_words']:.1f} |",
        f"| avg ground-truth words | {b['avg_gt_words']:.1f} | "
        f"{r['avg_gt_words']:.1f} |",
        "",
        "Caveat: lexical/semantic similarity against one ground-truth "
        "comment does NOT measure reviewer replication (many valid review "
        "comments are possible for one diff). Orientation numbers only.",
        ""]
    with outputs.open_run_file(run_dir, ".", "metrics_summary.md") as f:
        f.write("\n".join(lines))
    with outputs.open_run_file(run_dir, ".", "qualitative_examples.md") as f:
        f.write(qualitative_md(run_dir))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
