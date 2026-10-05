"""Four-mode review-generation pipeline (Base / Base+RAG / LoRA / LoRA+RAG).

The modes are deliberately SEPARATE — this is the research evaluation
structure; nothing collapses them into one pipeline:

  base      baseline_v1 template, no retrieval, no adapter
            (the validated generic local-LLM baseline)
  base_rag  rag_v1 template + frozen RAG retrieval
            (the validated RAG-conditioned mode)
  lora      sft_v1 template (the exact SFT training format) + reviewer
            adapter — no retrieval
  lora_rag  rag_v1 template + reviewer adapter + frozen RAG retrieval

Conceptual flow (modes with RAG): diff → reviewer → frozen retrieval
(top-3 reviewer-specific, cosine gate ≥ 0.5) → adapter selection →
generation → (optionally) displayed evidence. RAG examples are
inference-time evidence only; they are never training data.

The frozen RAG pipeline (reviewerbrain.inference.rag_retrieval) is a
CLIENT dependency here and is imported lazily so this module stays
importable without chromadb/sentence-transformers. `retriever` lets
callers (tests, UI) inject a substitute with the same retrieve(rec)
contract.
"""
from reviewerbrain.inference import adapters, prompts

MODES = ("base", "base_rag", "lora", "lora_rag")

MODE_TO_TEMPLATE = {
    "base": "baseline_v1",
    "base_rag": "rag_v1",
    "lora": "sft_v1",
    "lora_rag": "rag_v1",
}


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def mode_uses_rag(mode):
    check_mode(mode)
    return mode in ("base_rag", "lora_rag")


def mode_uses_adapter(mode):
    check_mode(mode)
    return mode in ("lora", "lora_rag")


def default_retriever():
    """The frozen serving retrieval (lazy import: chromadb + embeddings)."""
    from reviewerbrain.inference import rag_retrieval
    return rag_retrieval.retrieve


def run_review(mode, reviewer, rec, backend, retriever=None,
               adapters_root=None):
    """Generate one review in the given mode; returns a record dict.

    rec is a cleaned record (reviewerbrain.data.schema). The backend must
    expose generate(messages) -> (text, meta). Adapter availability is
    enforced BEFORE any retrieval or generation: a LoRA mode without the
    reviewer's adapter raises AdapterNotAvailable (train on Kaggle first,
    see kaggle/README.md).
    """
    check_mode(mode)
    if rec["reviewer"] != reviewer:
        raise ValueError(f"record reviewer {rec['reviewer']!r} != "
                         f"requested reviewer {reviewer!r}")
    if mode_uses_adapter(mode) and not adapters.has_adapter(reviewer,
                                                            adapters_root):
        raise adapters.AdapterNotAvailable(
            f"mode {mode!r} needs the reviewer adapter for {reviewer!r} — "
            f"none found (kaggle/README.md covers training and placement)")

    retrieved = None
    if mode_uses_rag(mode):
        retrieve = retriever if retriever is not None else default_retriever()
        retrieved = retrieve(rec)

    template = prompts.load_template(MODE_TO_TEMPLATE[mode])
    messages = prompts.build_messages(template, rec, reviewer, retrieved)
    text, meta = backend.generate(messages)
    return {
        "reviewer": reviewer,
        "mode": mode,
        "prompt_template": MODE_TO_TEMPLATE[mode],
        "uses_rag": mode_uses_rag(mode),
        "uses_adapter": mode_uses_adapter(mode),
        "messages": messages,
        "retrieved": retrieved,
        "generated_comment": text,
        "meta": meta,
    }
