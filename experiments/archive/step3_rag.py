"""Step 3 - ChromaDB RAG pipeline + cosine vs quantum-fidelity evaluation.

ARCHIVED — superseded by scripts/retrieval/build_indexes.py and
scripts/evaluation/run_heldout_eval.py. This is the historical v1
experiment (old hunk+comment document format, old processed/ layout).
Paths and behavior are period-correct and NOT maintained; the v1 indexes
it built were deleted. Kept for provenance of the 256-token truncation
finding (see docs/experiments/step3_rag_v1_truncation.md).

Builds three ChromaDB persistent indexes (thockin, ezyang, combined) from
the cleaned datasets using all-MiniLM-L6-v2 on CPU, with anonymized
telemetry disabled (no network egress beyond the one-time model download).

Indexed document = diff_hunk + newline + review_comment (follow_up_patch is
never indexed). Retrieval: top-3 with a 0.5 similarity gate.

Metric experiment: 5 deterministic dataset-derived queries against the
combined index. Both metrics use the SAME query embeddings and the SAME
document embedding matrix, over the SAME candidate universe (all docs minus
the query's own document):
  A. cosine similarity        (Chroma cosine space: sim = 1 - distance)
  B. quantum fidelity         (L2-normalized embeddings, |<a|b>|^2)
Gate-drop counts are computed over the full universe for BOTH metrics.

Writes NO files itself (ChromaDB persists its own index files); the full
markdown report is printed to stdout for shell redirection.
"""
import json
import os
import sys
import time

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLEAN = {
    "thockin": "D:/Projects/ReviewerBrain/ReviewBrainer/processed/thockin_clean.jsonl",
    "ezyang": "D:/Projects/ReviewerBrain/ReviewBrainer/processed/ezyang_clean.jsonl",
}
CHROMA_PATH = "D:/Projects/ReviewerBrain/ReviewBrainer/processed/chroma"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
GATE = 0.5
CANDIDATES = 10  # chroma fetch size (>= TOP_K)


def load_records():
    per_reviewer = {}
    for name, path in CLEAN.items():
        with open(path, "r", encoding="utf-8") as f:
            per_reviewer[name] = [json.loads(l) for l in f if l.strip()]
    return per_reviewer


def doc_text(r):
    return f"{r['diff_hunk']}\n\n{r['review_comment']}"


def meta_of(r):
    return {
        "reviewer": r["reviewer"],
        "pr_number": r["pr_number"],
        "pr_title": r["pr_title"] or "",
        "file": r["file"],
        "review_comment": r["review_comment"],
        "led_to_code_change": bool(r["led_to_code_change"]),
    }


def build_indexes(model, per_reviewer, out):
    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    for c in ("thockin", "ezyang", "combined"):
        try:
            client.delete_collection(c)
        except Exception:
            pass

    emb = {}
    for name, recs in per_reviewer.items():
        texts = [doc_text(r) for r in recs]
        t0 = time.perf_counter()
        e = model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True).astype(np.float32)
        dt = time.perf_counter() - t0
        emb[name] = e
        out.append(f"- indexed {name}: {len(recs)} docs, "
                   f"embedding took {dt:.1f}s ({1000*dt/len(recs):.1f} ms/doc)")

    all_recs = per_reviewer["thockin"] + per_reviewer["ezyang"]
    all_emb = np.vstack([emb["thockin"], emb["ezyang"]])
    ids = ([f"thockin:{i}" for i in range(len(per_reviewer["thockin"]))]
           + [f"ezyang:{i}" for i in range(len(per_reviewer["ezyang"]))])

    specs = [("thockin", per_reviewer["thockin"], emb["thockin"],
              [f"thockin:{i}" for i in range(len(per_reviewer["thockin"]))]),
             ("ezyang", per_reviewer["ezyang"], emb["ezyang"],
              [f"ezyang:{i}" for i in range(len(per_reviewer["ezyang"]))]),
             ("combined", all_recs, all_emb, ids)]

    collections = {}
    for cname, recs, e, cids in specs:
        col = client.create_collection(
            name=cname, metadata={"hnsw:space": "cosine"})
        for s in range(0, len(recs), 512):
            col.add(ids=cids[s:s+512], embeddings=e[s:s+512].tolist(),
                    documents=[doc_text(r) for r in recs[s:s+512]],
                    metadatas=[meta_of(r) for r in recs[s:s+512]])
        collections[cname] = col
        out.append(f"- chroma collection `{cname}`: {col.count()} docs")
    return client, collections, all_recs, all_emb, ids


def norm_rows(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def fmt_hits(hits, score_label, cos_of_id=None):
    if not hits:
        return ["  (no results passed the 0.5 gate)"]
    lines = []
    for rank, h in enumerate(hits, 1):
        m = h["meta"]
        extra = ""
        if cos_of_id is not None and h["id"] in cos_of_id:
            extra = f" (signed cos {cos_of_id[h['id']]:+.6f})"
        lines.append(
            f"  {rank}. [{h['id']}] PR#{m['pr_number']} `{m['file']}` "
            f"({m['reviewer']}) {score_label}={h['score']:.6f}{extra}")
    return lines


DEMO_QUERIES = {
    "thockin": [
        ("T1: new optional field added to a core API struct",
         "@@ -512,6 +512,11 @@ type ServiceSpec struct {\n"
         " \t// +optional\n"
         " \tSessionAffinityConfig *SessionAffinityConfig `json:\"sessionAffinityConfig,omitempty\"`\n"
         "+\n"
         "+\t// TrafficDistribution offers a hint for how traffic should be distributed\n"
         "+\t// among the endpoints of this service.\n"
         "+\t// +optional\n"
         "+\tTrafficDistribution *string `json:\"trafficDistribution,omitempty\" protobuf:\"bytes,16,opt,name=trafficDistribution\"`"),
        ("T2: validation logic added for an existing field",
         "@@ -204,8 +204,13 @@ func validateVolumes(vols []core.Volume, fldPath *field.Path) field.ErrorList {\n"
         " \t\t}\n"
         "+\t\tif vol.VolumeSource.EmptyDir != nil && vol.VolumeSource.EmptyDir.SizeLimit != nil {\n"
         "+\t\t\tif vol.VolumeSource.EmptyDir.SizeLimit.Sign() < 0 {\n"
         "+\t\t\t\tallErrs = append(allErrs, field.Invalid(idxPath.Child(\"emptyDir\").Child(\"sizeLimit\"),\n"
         "+\t\t\t\t\tvol.VolumeSource.EmptyDir.SizeLimit, \"must be greater than 0\"))\n"
         "+\t\t\t}\n"
         "+\t\t}"),
        ("T3: build script binary list change",
         "@@ -45,6 +45,9 @@ readonly KUBE_TEST_BINARIES=(\n"
         " )\n"
         "+readonly KUBE_TEST_SERVER_BINARIES=(\n"
         "+  kube-apiserver\n"
         "+  kube-controller-manager\n"
         "+)"),
    ],
    "ezyang": [
        ("E1: dynamo variable-tracking change",
         "@@ -712,6 +712,9 @@ class VariableTracker:\n"
         "     def var_getattr(self, obj, name):\n"
         "+        if name.startswith(\"_\"):\n"
         "+            raise Unsupported(\"accessing private attribute\")\n"
         "         guards = self.guards"),
        ("E2: C++ tensor concatenation size check",
         "@@ -3584,6 +3584,11 @@ void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension)\n"
         "+  int64_t outNumel = 0;\n"
         "+  for (int i = 0; i < numInputs; i++) {\n"
         "+    outNumel += THTensor_(nElement)(inputs[i]);\n"
         "+  }\n"
         "+  THArgCheck(outNumel >= 0, 2, \"invalid concatenation size\");"),
        ("E3: autograd test addition",
         "@@ -914,6 +914,9 @@ class TestAutograd(TestCase):\n"
         "     def test_autograd_simple(self):\n"
         "+        x = torch.randn(3, requires_grad=True)\n"
         "+        y = x * 2\n"
         "+        self.assertTrue(y.requires_grad)"),
    ],
}


def main():
    out = ["# ReviewerBrain — Step 3 RAG Report", ""]

    per_reviewer = load_records()
    out.append("## Index build")
    out.append("")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    out.append(f"- model: {MODEL_NAME} loaded on CPU in "
               f"{time.perf_counter()-t0:.1f}s, telemetry disabled")
    client, collections, all_recs, all_emb, ids = build_indexes(
        model, per_reviewer, out)
    dim = all_emb.shape[1]
    out.append(f"- embedding dimensions: {dim}")
    out.append("- document format: diff_hunk + newline + review_comment "
               "(follow_up_patch NOT indexed)")
    out.append(f"- combined index metadata includes reviewer identity; "
               f"total corpus: {len(all_recs)} docs")
    out.append("")

    metas = [meta_of(r) for r in all_recs]
    doc_emb_normed = norm_rows(all_emb)
    row_of = {doc_id: i for i, doc_id in enumerate(ids)}
    universe = len(all_recs) - 1  # full corpus minus the query's own doc

    # --- metric experiment: 5 deterministic queries against combined ---
    out.append("## Metric experiment — cosine vs quantum fidelity")
    out.append("")
    out.append(f"5 queries sampled deterministically (no RNG) from "
               f"combined_clean.jsonl: 3 thockin at 10%/50%/90%, 2 ezyang at "
               f"30%/70% of each reviewer's block. Query = that example's "
               f"diff_hunk; its own document is excluded from both metrics. "
               f"Identical embeddings for both metrics; only the ranking "
               f"metric differs. Candidate universe for both: {universe} docs "
               f"(full corpus minus self). Gate: score >= 0.5 kept. Scores "
               f"shown to 6 decimals.")
    out.append("")

    n_t = len(per_reviewer["thockin"])
    picks = ([("thockin", int(0.1 * n_t)), ("thockin", int(0.5 * n_t)),
              ("thockin", int(0.9 * n_t)),
              ("ezyang", int(0.3 * len(per_reviewer["ezyang"]))),
              ("ezyang", int(0.7 * len(per_reviewer["ezyang"])))])

    embed_ms, chroma_ms, fid_ms = [], [], []
    overlap_counts = []
    rank_changed = 0
    cos_dropped_total = fid_dropped_total = 0
    cos_shown = fid_shown = 0
    max_sim_diff = 0.0

    for qi, (rev, local_i) in enumerate(picks, 1):
        rec = per_reviewer[rev][local_i]
        qid = f"{rev}:{local_i}"
        self_row = row_of[qid]
        qtext = rec["diff_hunk"]

        t0 = time.perf_counter()
        qemb = model.encode(qtext, convert_to_numpy=True).astype(np.float32)
        embed_ms.append((time.perf_counter() - t0) * 1000)

        out.append(f"### Query {qi} — from {qid} (PR#{rec['pr_number']} "
                   f"`{rec['file']}`, {rev})")
        out.append("")
        out.append("```diff")
        excerpt = qtext if len(qtext) <= 900 else qtext[:900] + "\n… (truncated)"
        out.append(excerpt)
        out.append("```")
        out.append("")

        # full-corpus scores from the SAME embeddings (for gate accounting)
        qn = qemb / max(float(np.linalg.norm(qemb)), 1e-12)
        cos_all = doc_emb_normed @ qn          # signed cosine, all docs
        fid_all = np.abs(cos_all) ** 2          # |<a|b>|^2

        # metric A: chroma (production path)
        t0 = time.perf_counter()
        res = collections["combined"].query(
            query_embeddings=[qemb.tolist()], n_results=CANDIDATES,
            include=["distances", "metadatas"])
        chroma_ms.append((time.perf_counter() - t0) * 1000)
        cos_hits = []
        for i, doc_id in enumerate(res["ids"][0]):
            if doc_id == qid:
                continue
            sim = 1.0 - res["distances"][0][i]
            if sim >= GATE:
                cos_hits.append({"id": doc_id, "score": sim,
                                 "meta": res["metadatas"][0][i]})
            if len(cos_hits) == TOP_K:
                break
        cos_dropped = int((np.delete(cos_all, self_row) < GATE).sum())
        cos_dropped_total += cos_dropped
        cos_shown += len(cos_hits)

        # metric B: fidelity full scan
        t0 = time.perf_counter()
        fid_masked = fid_all.copy()
        fid_masked[self_row] = -1.0
        order = np.argsort(-fid_masked)
        fid_hits = []
        for idx in order:
            if fid_masked[idx] < GATE:
                break
            fid_hits.append({"id": ids[idx], "score": float(fid_masked[idx]),
                             "meta": metas[idx]})
            if len(fid_hits) == TOP_K:
                break
        fid_ms.append((time.perf_counter() - t0) * 1000)
        fid_dropped = int((fid_masked[fid_masked > -1] < GATE).sum())
        fid_dropped_total += fid_dropped
        fid_shown += len(fid_hits)

        # cross-check chroma cosine vs numpy on chroma's top hits
        for h in cos_hits[:3]:
            np_sim = float(cos_all[row_of[h["id"]]])
            max_sim_diff = max(max_sim_diff, abs(np_sim - h["score"]))

        cos_set = {h["id"] for h in cos_hits}
        fid_set = {h["id"] for h in fid_hits}
        ov = len(cos_set & fid_set)
        overlap_counts.append(ov)
        changed = ([h["id"] for h in cos_hits] != [h["id"] for h in fid_hits])
        rank_changed += changed

        out.append(f"**Cosine top-3** (universe {cos_dropped + (universe - cos_dropped)} = "
                   f"{universe}; dropped by gate: {cos_dropped}):")
        out.extend(fmt_hits(cos_hits, "cos"))
        out.append("")
        out.append(f"**Fidelity |<a|b>|² top-3** (universe {universe}; "
                   f"dropped by gate: {fid_dropped}):")
        out.extend(fmt_hits(
            fid_hits, "fid",
            cos_of_id={h["id"]: float(cos_all[row_of[h["id"]]])
                       for h in fid_hits}))
        out.append("")
        out.append(f"- top-3 set overlap: {ov}/3 {sorted(cos_set & fid_set)}")
        out.append(f"- ranking changed: {'YES' if changed else 'no'}")
        out.append("")

    out += [
        "## Per-reviewer retrieval demos (cosine metric)",
        "",
        "3 handcrafted synthetic mini-diffs per reviewer, run against that "
        "reviewer's own index. These are new diffs, so no self-exclusion "
        "applies; the 0.5 gate is active.",
        "",
    ]
    for rev, queries in DEMO_QUERIES.items():
        for label, qtext in queries:
            t0 = time.perf_counter()
            qemb = model.encode(qtext, convert_to_numpy=True).astype(np.float32)
            embed_ms.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            res = collections[rev].query(
                query_embeddings=[qemb.tolist()], n_results=CANDIDATES,
                include=["distances", "metadatas"])
            chroma_ms.append((time.perf_counter() - t0) * 1000)
            all_hits = []
            for i, doc_id in enumerate(res["ids"][0]):
                sim = 1.0 - res["distances"][0][i]
                all_hits.append((sim, doc_id, res["metadatas"][0][i]))
            all_hits.sort(key=lambda x: -x[0])
            hits = [{"id": d, "score": s, "meta": m}
                    for s, d, m in all_hits if s >= GATE][:TOP_K]
            dropped = sum(1 for s, _, _ in all_hits if s < GATE)
            out.append(f"### {label} — index `{rev}` (top-{CANDIDATES} fetched, "
                       f"dropped by gate: {dropped})")
            out.append("```diff")
            out.append(qtext)
            out.append("```")
            if not hits:
                best = all_hits[0]
                out.append(f"**No results passed the 0.5 gate.** Best "
                           f"candidate: [{best[1]}] PR#{best[2]['pr_number']} "
                           f"cos={best[0]:.6f}")
            for rank, h in enumerate(hits, 1):
                m = h["meta"]
                c = m["review_comment"]
                c = c if len(c) <= 300 else c[:300] + "…"
                out.append(f"- {rank}. PR#{m['pr_number']} `{m['file']}` "
                           f"cos={h['score']:.6f}")
                out.append(f"  > {c}")
            out.append("")

    out += [
        "## Latency (CPU, this machine)",
        "",
        f"- query embedding (MiniLM, single text): avg "
        f"{sum(embed_ms)/len(embed_ms):.1f} ms over {len(embed_ms)} queries",
        f"- chroma cosine query (top-{CANDIDATES} fetch): avg "
        f"{sum(chroma_ms)/len(chroma_ms):.1f} ms over {len(chroma_ms)} queries",
        f"- fidelity full-scan scoring (numpy, {len(all_recs)}×{dim}): avg "
        f"{sum(fid_ms)/len(fid_ms):.2f} ms over {len(fid_ms)} queries",
        "",
        "## Aggregates",
        "",
        f"- cosine vs fidelity top-3 set overlap per experiment query: "
        f"{overlap_counts} → mean {sum(overlap_counts)/len(overlap_counts):.1f}/3",
        f"- experiment queries where the ordered top-3 changed: "
        f"{rank_changed}/5",
        f"- gate drops in experiment (universe {universe} per query): "
        f"cosine {cos_dropped_total} across 5 queries; fidelity "
        f"{fid_dropped_total}",
        f"- experiment result slots filled: cosine {cos_shown}/15, "
        f"fidelity {fid_shown}/15",
        f"- chroma cosine vs numpy cosine max abs difference: "
        f"{max_sim_diff:.2e}",
        "",
    ]

    print("\n".join(out))


if __name__ == "__main__":
    main()
