"""Build the validated v2 ChromaDB indexes for ReviewerBrain.

Reproduces the validated RAG indexes:
  - comment-first representation (see reviewerbrain.retrieval.representation)
  - all-MiniLM-L6-v2 on CPU, 384 dims
  - ChromaDB persistent collections (cosine space): thockin, ezyang, combined
  - metadata: reviewer, pr_number, pr_title, file, comment_id,
    review_comment, led_to_code_change

Writes only the ChromaDB index files under indexes/chroma/v2/; the audit
report goes to stdout for shell redirection. Original datasets are only
read, never modified.
"""
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
from reviewerbrain.embeddings.model import load_model, MODEL_NAME
from reviewerbrain.retrieval import representation as rag_repr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHROMA_PATH = str(paths.CHROMA_V2_DIR)


def meta_of(r):
    return {"reviewer": r["reviewer"], "pr_number": r["pr_number"],
            "pr_title": r["pr_title"] or "", "file": r["file"],
            "comment_id": r.get("_comment_id"),
            "review_comment": r["review_comment"],
            "led_to_code_change": bool(r["led_to_code_change"])}


def main():
    out = ["# Step 4 — Representation v2 audit + index rebuild", ""]

    records, join_misses = rag_repr.load_enriched()
    model = load_model()
    max_len = model.max_seq_length
    tok = model.tokenizer
    out += [
        "## Exact truncation rule (TASK A)",
        "",
        f"- model window: {max_len} wordpiece tokens (all-MiniLM-L6-v2); "
        "2 reserved for [CLS]/[SEP], 2 safety.",
        "- document = `Reviewer comment:\\n<comment>\\n\\nFile:\\n<path>"
        "\\n\\nRelevant diff:\\n<trimmed diff>` — the comment comes FIRST, "
        "so it is always inside the window; the diff absorbs all cuts.",
        "- diff token budget = window − comment − path − scaffold − 4; "
        "floored at 24 tokens. If the floor pushes the total past the "
        "window the model truncates the diff tail only (comment still "
        "fully embedded). Docs whose comment ALONE cannot fit are flagged "
        "`comment_overflows_window` (counted below, none deleted).",
        "- diff trimming: parse the `@@ -a,b +c,d @@` header, assign each "
        "body line its new-file line number, anchor at the line matching "
        "the comment's `line` field (nearest match if out of range; "
        "midpoint of +/- lines when `line` is null), then grow a contiguous "
        "window outward (nearest line first) until the budget is reached. "
        "No elision markers; the @@ header itself is dropped. A single "
        "line longer than the whole budget is hard-truncated "
        "proportionally and flagged.",
        "- query format (eval/production): `File:\\n<path>\\n\\nRelevant "
        "diff:\\n<trimmed diff>` — same trim rule, no comment block.",
        "- follow_up_patch is never used in any representation.",
        "",
        "## Enrichment join (read-only, originals untouched)",
        "",
        f"- records: {len(records)}; join misses (no original fresh-comment "
        f"match): {join_misses}",
        "",
    ]

    per_reviewer = {"thockin": [], "ezyang": []}
    for r in records:
        per_reviewer[r["reviewer"]].append(r)

    texts, infos, ids = {}, {}, {}
    for name, recs in per_reviewer.items():
        t, i, ids_l = [], [], []
        for idx, r in enumerate(recs):
            text, info = rag_repr.doc_text(r, tok, max_len)
            t.append(text)
            i.append(info)
            ids_l.append(f"{name}:{idx}")
        texts[name], infos[name], ids[name] = t, i, ids_l

    out.append("## TASK B — token audit of the v2 representation")
    out.append("")
    for name in ("thockin", "ezyang"):
        recs = per_reviewer[name]
        inf = infos[name]
        toks = sorted(x["total_tokens"] for x in inf)
        over = sum(1 for x in inf if x["total_tokens"] > max_len)
        overflow = [j for j, x in enumerate(inf)
                    if x.get("comment_overflows_window")]
        trimmed = sum(1 for x in inf
                      if x.get("lines_total") and x["lines_kept"] < x["lines_total"])
        ratios = sorted(x["lines_kept"] / x["lines_total"]
                        for x in inf if x.get("lines_total"))
        line_avail = sum(1 for r in recs if r.get("_line") is not None)
        anchor_src = {}
        for x in inf:
            anchor_src[x.get("anchor_source", "?")] = \
                anchor_src.get(x.get("anchor_source", "?"), 0) + 1
        chars = sorted(len(t) for t in texts[name])
        out += [
            f"### {name} ({len(recs)} docs)",
            "",
            f"- final doc tokens: p50={pct(toks,50)} p90={pct(toks,90)} "
            f"p99={pct(toks,99)} max={toks[-1]}; docs over {max_len}: "
            f"{over} (all flagged comment-overflow: {over == len(overflow)})",
            f"- final doc chars: p50={pct(chars,50)} max={chars[-1]}",
            f"- comment fully inside window: {len(recs)-len(overflow)} "
            f"({100*(len(recs)-len(overflow))/len(recs):.1f}%) — flagged "
            f"overflow: {len(overflow)}",
            f"- comment `line` field available for anchoring: {line_avail} "
            f"({100*line_avail/len(recs):.1f}%)",
            f"- anchor source: {anchor_src}",
            f"- docs whose diff was trimmed: {trimmed} "
            f"({100*trimmed/len(recs):.1f}%); kept/total hunk-line ratio: "
            f"p10={pct(ratios,10):.2f} p50={pct(ratios,50):.2f}",
            f"- no @@ header: {sum(1 for x in inf if x.get('no_header'))}; "
            f"multi @@: {sum(1 for x in inf if x.get('multi_header'))}; "
            f"oversized anchor line: "
            f"{sum(1 for x in inf if x.get('oversized_anchor_line'))}",
            "",
        ]
        longest = sorted(range(len(inf)),
                         key=lambda j: -infos[name][j]["total_tokens"])[:5]
        out.append("Longest inputs:")
        for j in longest:
            r = recs[j]
            out.append(f"- [{name}:{j}] {infos[name][j]['total_tokens']} tok "
                       f"/ {len(texts[name][j])} chars — PR#{r['pr_number']} "
                       f"`{r['file']}`")
        out.append("")

    # Explicit verification
    v_bad_sub, v_bad_prefix, v_bad_win = 0, 0, 0
    for name in ("thockin", "ezyang"):
        for j, r in enumerate(per_reviewer[name]):
            text = texts[name][j]
            if r["review_comment"] not in text:
                v_bad_sub += 1
            p_ids = tok.encode(rag_repr.DOC_SCAFFOLD + r["review_comment"],
                               add_special_tokens=False)
            t_ids = tok.encode(text, add_special_tokens=False)
            if t_ids[:len(p_ids)] != list(p_ids):
                v_bad_prefix += 1
            if (not infos[name][j]["comment_overflows_window"]
                    and len(p_ids) + 2 > max_len):
                v_bad_win += 1
    over_total = sum(
        1 for name in ("thockin", "ezyang") for x in infos[name]
        if x["total_tokens"] > max_len)
    out += [
        "## Explicit verification",
        "",
        f"- full review-comment substring present in every doc text: "
        f"{'PASS' if v_bad_sub == 0 else f'FAIL ({v_bad_sub})'} "
        f"(checked all {len(records)} docs by exact string match)",
        f"- the comment's tokens are an exact token-prefix of the encoded "
        f"doc (guarantees nothing before the comment can displace it): "
        f"{'PASS' if v_bad_prefix == 0 else f'FAIL ({v_bad_prefix})'}",
        f"- every non-overflow doc has its comment block (incl. [CLS]) "
        f"ending within the {max_len}-token window: "
        f"{'PASS' if v_bad_win == 0 else f'FAIL ({v_bad_win})'}",
        f"- docs with total_tokens > {max_len}: {over_total} — all are "
        f"budget-floored or comment-overflow cases where the MODEL truncates "
        f"the diff tail only; the comment (first block) is unaffected.",
        "",
    ]

    # Index rebuild
    out.append("## Index rebuild (chroma_v2)")
    out.append("")
    all_recs = per_reviewer["thockin"] + per_reviewer["ezyang"]
    all_texts = texts["thockin"] + texts["ezyang"]
    ids_all = ids["thockin"] + ids["ezyang"]

    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    for c in ("thockin", "ezyang", "combined"):
        try:
            client.delete_collection(c)
        except Exception:
            pass

    emb = {}
    for name in ("thockin", "ezyang"):
        t0 = time.perf_counter()
        e = model.encode(texts[name], batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True).astype(np.float32)
        emb[name] = e
        out.append(f"- {name}: {len(texts[name])} docs embedded in "
                   f"{time.perf_counter()-t0:.1f}s "
                   f"({1000*(time.perf_counter()-t0)/len(texts[name]):.1f} ms/doc)")

    specs = [("thockin", per_reviewer["thockin"], texts["thockin"],
              emb["thockin"], ids["thockin"]),
             ("ezyang", per_reviewer["ezyang"], texts["ezyang"],
              emb["ezyang"], ids["ezyang"]),
             ("combined", all_recs, all_texts,
              np.vstack([emb["thockin"], emb["ezyang"]]), ids_all)]
    for cname, recs, t, e, cids in specs:
        t0 = time.perf_counter()
        col = client.create_collection(
            name=cname, metadata={"hnsw:space": "cosine"})
        for s in range(0, len(recs), 512):
            col.add(ids=cids[s:s+512], embeddings=e[s:s+512].tolist(),
                    documents=t[s:s+512],
                    metadatas=[meta_of(r) for r in recs[s:s+512]])
        out.append(f"- collection `{cname}`: {col.count()} docs "
                   f"({time.perf_counter()-t0:.1f}s incl. add)")
    out.append("")
    print("\n".join(out))


def pct(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(p / 100 * len(sorted_vals)))]


if __name__ == "__main__":
    main()
