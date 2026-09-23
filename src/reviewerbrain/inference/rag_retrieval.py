"""Top-k reviewer retrieval for generation — a thin CLIENT of the validated
RAG pipeline.

Nothing here changes the pipeline: the query text is produced by the frozen
`reviewerbrain.retrieval.representation.query_text`, embedded by the frozen
`reviewerbrain.embeddings.model.load_model`, and searched against the
existing per-reviewer ChromaDB collection (frozen serving config:
per-reviewer index, cosine, top-k=3, gate cosine >= 0.5).

LEAKAGE GUARD: the persistent indexes contain every cleaned example,
INCLUDING held-out PRs — the validated held-out evaluation excluded held-out
PR rows at query time (`scripts/evaluation/run_heldout_eval.py`, idx_rows),
and so does this module: candidates whose pr_number is held out for the
reviewer (frozen deterministic split, reviewerbrain.evaluation.split) are
dropped before the gate. Every result is re-checked against the query's own
PR and the check is recorded in the output.
"""
import json

import chromadb
from chromadb.config import Settings

from reviewerbrain import paths
from reviewerbrain.embeddings.model import load_model
from reviewerbrain.evaluation.split import held_out_prs
from reviewerbrain.retrieval import representation as rag_repr

# Frozen serving constants (mirrored in configs/rag/default.yaml; tests
# assert the two stay in sync — same convention as the retrieval scripts).
TOP_K = 3
GATE_COS = 0.5
N_CANDIDATES = 30   # fetched before held-PR filtering + gate

_state = {}


def _client():
    if "client" not in _state:
        _state["client"] = chromadb.PersistentClient(
            path=str(paths.CHROMA_V2_DIR),
            settings=Settings(anonymized_telemetry=False))
    return _state["client"]


def _encoder():
    """(model, tokenizer, max_seq_length), loaded once."""
    if "model" not in _state:
        model = load_model()
        _state["model"] = model
        _state["tok"] = model.tokenizer
        _state["max_len"] = model.max_seq_length
    return _state["model"], _state["tok"], _state["max_len"]


def held_prs():
    """Frozen held-out PR sets per reviewer, computed once from the cleaned
    datasets via the validated split rule (read-only)."""
    if "held_prs" not in _state:
        rows = []
        for name, path in paths.CLEAN_FILES.items():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        rows.append({"reviewer": r["reviewer"],
                                     "pr_number": r["pr_number"]})
        _state["held_prs"] = held_out_prs(rows)
    return _state["held_prs"]


def retrieve(rec):
    """Frozen-pipeline retrieval for one cleaned record, held-PR-filtered.

    Returns a list of up to TOP_K dicts: {doc_id, cos, file, pr_number,
    comment, document} (document = the validated comment-first index text,
    inserted verbatim into the RAG prompt). Candidates from held-out PRs
    are excluded (leakage guard), then the frozen cosine gate applies.
    """
    model, tok, max_len = _encoder()
    q_text, _ = rag_repr.query_text(rec, tok, max_len)
    q = model.encode([q_text], batch_size=1, show_progress_bar=False,
                     convert_to_numpy=True)[0].tolist()
    banned = held_prs()[rec["reviewer"]]
    col = _client().get_collection(rec["reviewer"])
    out = []
    n_fetch = N_CANDIDATES
    while True:
        got = col.query(query_embeddings=[q], n_results=n_fetch,
                        include=["documents", "metadatas", "distances"])
        out = []
        for doc_id, doc, meta, dist in zip(got["ids"][0],
                                           got["documents"][0],
                                           got["metadatas"][0],
                                           got["distances"][0]):
            if meta.get("pr_number") in banned:
                continue    # leakage guard: held-out PR rows are not candidates
            cos = 1.0 - float(dist)   # cosine space: chroma distance = 1 - cos
            if cos < GATE_COS:
                break     # distance-sorted; nothing further passes the gate
            out.append({"doc_id": doc_id, "cos": cos,
                        "file": meta.get("file"),
                        "pr_number": meta.get("pr_number"),
                        "comment": meta.get("review_comment"),
                        "document": doc})
            if len(out) == TOP_K:
                break
        if len(out) == TOP_K or n_fetch >= col.count():
            break
        n_fetch = min(n_fetch * 2, col.count())   # re-fetch wider if the
        # held-PR filter consumed too many candidates
    assert all(r["pr_number"] != rec["pr_number"] for r in out), \
        "leak: retrieved a candidate from the query's own PR"
    return out


def close():
    """Drop cached handles (chroma/sqlite locks on Windows)."""
    _state.clear()
