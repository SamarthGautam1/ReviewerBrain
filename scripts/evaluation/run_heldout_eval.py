"""Held-out evaluation of the ReviewerBrain v2 RAG representation.

Deterministic PR-level split (see reviewerbrain.evaluation.split): per
reviewer, every 5th PR (sorted by pr_number, position % 5 == 4) is held
out. Queries = held-out examples; the retrieval index contains NO example
from any held-out PR, so neither the query itself nor same-PR siblings can
be retrieved. No RNG anywhere.

Compares cosine vs fidelity = cos^2 with EQUIVALENT thresholds
(cos >= 0.5 <=> fid >= 0.25) plus a no-threshold run, over the SAME
embeddings. Writes NO files itself; stdout has two sections:
  ===QUERIES===  per-query eval records (JSONL)
  ===REPORT===   full markdown report
Redirect each section to its target file (see
docs/methodology/rag_reproduction.md).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import numpy as np
import chromadb
from chromadb.config import Settings

from reviewerbrain import paths
from reviewerbrain.embeddings.model import load_model
from reviewerbrain.evaluation import metrics
from reviewerbrain.evaluation.split import held_out_prs
from reviewerbrain.retrieval import representation as rag_repr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THRESH_COS = 0.5
THRESH_FID = 0.25  # equivalent: cos >= 0.5  <=>  cos^2 >= 0.25
KS = (1, 3, 5)


def main():
    records, join_misses = rag_repr.load_enriched()
    n_t = sum(1 for r in records if r["reviewer"] == "thockin")
    ids = ([f"thockin:{i}" for i in range(n_t)]
           + [f"ezyang:{i}" for i in range(len(records) - n_t)])

    client = chromadb.PersistentClient(
        path=str(paths.CHROMA_V2_DIR),
        settings=Settings(anonymized_telemetry=False))
    got = client.get_collection("combined").get(include=["embeddings"])
    assert got["ids"] == ids, "chroma_v2 id order mismatch"
    E = np.array(got["embeddings"], dtype=np.float32)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    held_prs = held_out_prs(records)

    is_query = np.array([r["pr_number"] in held_prs[r["reviewer"]]
                         for r in records])
    q_rows = np.where(is_query)[0]
    idx_rows = np.where(~is_query)[0]
    dirs = [os.path.dirname(r["file"] or "") for r in records]

    model = load_model()
    max_len = model.max_seq_length
    tok = model.tokenizer

    q_texts = []
    for row in q_rows:
        t, _ = rag_repr.query_text(records[row], tok, max_len)
        q_texts.append(t)
    t0 = time.perf_counter()
    Q = model.encode(q_texts, batch_size=64, show_progress_bar=False,
                     convert_to_numpy=True).astype(np.float32)
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    enc_s = time.perf_counter() - t0

    def fresh_agg():
        return {"n": 0,
                "cos": {"hit": {k: 0 for k in KS},
                        "dirhit": {k: 0 for k in KS},
                        "sim": {k: 0.0 for k in KS},
                        "thit": {k: 0 for k in KS}},
                "fid": {"hit": {k: 0 for k in KS},
                        "dirhit": {k: 0 for k in KS},
                        "sim": {k: 0.0 for k in KS},
                        "thit": {k: 0 for k in KS}},
                "overlap": {k: 0 for k in KS},
                "spearman": [], "signfold": 0,
                "cos_zero": 0, "fid_zero": 0, "leak": 0,
                "mode_top1_changes": 0}

    agg = {"per_reviewer": fresh_agg(), "combined": fresh_agg()}
    rec_out = []

    for qi, row in enumerate(q_rows):
        r = records[row]
        cos_all = E @ Q[qi]
        fid_all = cos_all ** 2
        rec_out_row = None
        mode_top1 = {}

        for mode in ("per_reviewer", "combined"):
            mask = (idx_rows == idx_rows)  # all True
            if mode == "per_reviewer":
                mask = np.array([records[i]["reviewer"] == r["reviewer"]
                                 for i in idx_rows])
            cand = idx_rows[mask]
            a = agg[mode]
            a["n"] += 1

            c = cos_all[cand]          # local scores
            f = fid_all[cand]
            c_ord = np.argsort(-c)     # local positions, cosine order
            f_ord = np.argsort(-f)     # local positions, fidelity order
            a["spearman"].append(metrics.spearman(c, f))
            a["signfold"] += int((c[f_ord[:5]] < 0).any())

            for metric, ord_local, scores, thr in (
                    ("cos", c_ord, c, THRESH_COS),
                    ("fid", f_ord, f, THRESH_FID)):
                top = cand[ord_local[:KS[-1]]]      # global rows, top-5
                kept_local = ord_local[scores[ord_local] >= thr]
                if metric == "cos":
                    a["cos_zero"] += 1 if len(kept_local) == 0 else 0
                else:
                    a["fid_zero"] += 1 if len(kept_local) == 0 else 0
                for k in KS:
                    topk = top[:k]
                    a[metric]["hit"][k] += int(any(
                        records[i]["file"] == r["file"] for i in topk))
                    a[metric]["dirhit"][k] += int(any(
                        dirs[i] == dirs[row] for i in topk))
                    a[metric]["sim"][k] += float(np.mean(
                        scores[ord_local[:k]]))
                    kept_top = cand[kept_local[:k]]
                    a[metric]["thit"][k] += int(any(
                        records[i]["file"] == r["file"] for i in kept_top))
                if metric == "cos":
                    a["leak"] += int(any(
                        records[i]["pr_number"] == r["pr_number"]
                        for i in top))

            for k in KS:
                a["overlap"][k] += len(
                    {int(i) for i in cand[c_ord[:k]]}
                    & {int(i) for i in cand[f_ord[:k]]})
            mode_top1[mode] = int(cand[c_ord[0]])

        if (len(mode_top1) == 2
                and mode_top1["per_reviewer"] != mode_top1["combined"]):
            agg["combined"]["mode_top1_changes"] += 1

        # saved record: combined-index cosine ranking, no threshold, top-5
        c_all = cos_all[idx_rows]
        top5 = idx_rows[np.argsort(-c_all)[:5]]
        rel = []
        for rank, i in enumerate(top5, 1):
            i = int(i)
            rel.append({
                "rank": rank, "doc_id": ids[i],
                "reviewer": records[i]["reviewer"],
                "pr_number": records[i]["pr_number"],
                "file": records[i]["file"],
                "cos": float(cos_all[i]), "fid": float(fid_all[i]),
                "same_file": records[i]["file"] == r["file"],
                "same_dir": dirs[i] == dirs[row],
                "review_comment": records[i]["review_comment"]})
        rec_out_row = {
            "query_id": ids[row],
            "query_pr_number": r["pr_number"],
            "query_comment_id": r.get("_comment_id"),
            "reviewer": r["reviewer"],
            "file": r["file"],
            "query_diff_trimmed": q_texts[qi],
            "query_review_comment": r["review_comment"],
            "retrieved": rel}
        rec_out.append(rec_out_row)

    print("===QUERIES===")
    for rec in rec_out:
        print(json.dumps(rec, ensure_ascii=False))

    out = ["# Step 4 — Held-out RAG evaluation (v2 representation)", ""]

    held_counts = {name: sum(1 for r in records if r["reviewer"] == name
                             and r["pr_number"] in held_prs[name])
                   for name in ("thockin", "ezyang")}
    idx_counts = {name: sum(1 for r in records if r["reviewer"] == name
                            and r["pr_number"] not in held_prs[name])
                  for name in ("thockin", "ezyang")}
    out += [
        "## TASK D — split definition",
        "",
        "- deterministic: per reviewer, PRs sorted by pr_number, every 5th "
        "PR (position % 5 == 4) held out. No RNG; reruns reproduce the "
        "split exactly.",
        f"- thockin: {len(held_prs['thockin'])} held-out PRs → "
        f"{held_counts['thockin']} queries; index {idx_counts['thockin']} docs",
        f"- ezyang: {len(held_prs['ezyang'])} held-out PRs → "
        f"{held_counts['ezyang']} queries; index {idx_counts['ezyang']} docs",
        f"- total queries: {len(q_rows)}; query embedding time {enc_s:.1f}s; "
        f"index rows exclude every held-out PR (same-PR leakage impossible "
        f"by construction; verified: {agg['combined']['leak'] + agg['per_reviewer']['leak']} leaks in top-5 across all queries/modes)",
        "- relevance proxies (objective): same-file hit = retrieved file "
        "== query file; same-dir hit = same parent directory. Semantic "
        "relevance is assessed qualitatively (see evaluations/reports).",
        "- gated hit@k counts a miss when nothing passes the gate.",
        "",
    ]

    out.append("## TASK E — cosine vs fidelity (equivalent thresholds)")
    out.append("")
    out.append("Gates: cosine >= 0.5 vs fidelity >= 0.25 (equivalent "
               "admission for positive cosines). 'no thresh' rows use "
               "pure ranking. Mean-score rows report each metric's own "
               "score over its own top-k.")
    out.append("")
    for mode in ("per_reviewer", "combined"):
        a = agg[mode]
        n = a["n"]
        out.append(f"### mode: {mode} — {n} queries")
        out.append("")
        out.append("| metric | cosine | fidelity |")
        out.append("|---|---|---|")
        for k in KS:
            out.append(f"| same-file hit@{k} (no thresh) | "
                       f"{100*a['cos']['hit'][k]/n:.1f}% | "
                       f"{100*a['fid']['hit'][k]/n:.1f}% |")
            out.append(f"| same-dir hit@{k} (no thresh) | "
                       f"{100*a['cos']['dirhit'][k]/n:.1f}% | "
                       f"{100*a['fid']['dirhit'][k]/n:.1f}% |")
            out.append(f"| mean score of top-{k} (no thresh) | "
                       f"{a['cos']['sim'][k]/n:.4f} | "
                       f"{a['fid']['sim'][k]/n:.4f} |")
            out.append(f"| same-file hit@{k} (gated) | "
                       f"{100*a['cos']['thit'][k]/n:.1f}% | "
                       f"{100*a['fid']['thit'][k]/n:.1f}% |")
        out.append(f"| zero-candidate queries (gated) | {a['cos_zero']} | "
                   f"{a['fid_zero']} |")
        out.append("")
        out.append(f"- top-k set overlap cosine∩fidelity (no thresh): "
                   + ", ".join(f"@{k}={a['overlap'][k]/n:.2f}/{k}"
                               for k in KS))
        sp = [x for x in a["spearman"] if x == x]
        out.append(f"- mean Spearman rank correlation cos vs fid over full "
                   f"candidate lists: {sum(sp)/len(sp):.4f} "
                   f"(1.0 = identical ordering)")
        out.append(f"- queries where a fidelity top-5 member has negative "
                   f"cosine (sign-fold fired): {a['signfold']}")
        if mode == "combined":
            out.append(f"- queries where top-1 differs between the "
                       f"per_reviewer and combined index: "
                       f"{a['mode_top1_changes']}")
        out.append("")

    out += [
        "## TASK F — qualitative inspection (held-out queries)",
        "",
        "10 evenly-spaced held-out queries per reviewer, top-3 by cosine "
        "over the combined held-out index (no threshold). Flags: [DIR] "
        "same directory, [BOILER] license/boilerplate head in a hunk, "
        "[WEAK] cos < 0.3, [LEAK] same PR (expected 0).",
        "",
    ]
    q_pos = {row: qi for qi, row in enumerate(q_rows)}
    for name in ("thockin", "ezyang"):
        rows = [row for row in q_rows if records[row]["reviewer"] == name]
        picks = [int(rows[i * len(rows) // 10]) for i in range(10)]
        out.append(f"### {name} — 10 held-out queries")
        out.append("")
        for row in picks:
            r = records[row]
            qi = q_pos[row]
            cos_all = E[idx_rows] @ Q[qi]
            order = np.argsort(-cos_all)[:3]
            out.append(f"#### {ids[row]} — PR#{r['pr_number']} "
                       f"`{r['file']}` (comment_id {r.get('_comment_id')})")
            c = r["review_comment"]
            out.append(f"- **Query review:** "
                       f"{c[:400] + ('…' if len(c) > 400 else '')}")
            qd = q_texts[qi]
            qd = qd[qd.find("Relevant diff:\n") + len("Relevant diff:\n"):]
            qd = qd[:600] + ("… (trimmed)" if len(qd) > 600 else "")
            out.append("```diff")
            out.append(qd)
            out.append("```")
            qb = metrics.is_boilerplate(r["diff_hunk"])
            for rank, li in enumerate(order, 1):
                i = int(idx_rows[li])
                h = records[i]
                flags = []
                if dirs[i] == dirs[row]:
                    flags.append("DIR")
                if metrics.is_boilerplate(h["diff_hunk"]) or qb:
                    flags.append("BOILER")
                if cos_all[li] < 0.3:
                    flags.append("WEAK")
                if h["pr_number"] == r["pr_number"]:
                    flags.append("LEAK")
                hc = h["review_comment"]
                hc = hc[:280] + ("…" if len(hc) > 280 else "")
                out.append(f"- {rank}. {ids[i]} PR#{h['pr_number']} "
                           f"`{h['file']}` cos={cos_all[li]:.4f} "
                           f"{'['+','.join(flags)+']' if flags else ''}")
                out.append(f"  > {hc}")
            out.append("")

    print("===REPORT===")
    print("\n".join(out))


if __name__ == "__main__":
    main()
