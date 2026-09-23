# ReviewerBrain

ReviewerBrain replicates the review behavior of a specific GitHub code
reviewer — not a generic reviewer — by converting that person's historical
PR review comments into a reviewer-specific retrieval knowledge base (RAG),
and later fine-tuning a local LLM on the same data. Everything runs locally:
no cloud APIs, no data egress.

## Problem

Senior maintainers review new code with strong personal conventions: what
they look at first, what they always ask about, how terse or verbose they
are. Generic LLM review does not capture any individual's judgment. The
question ReviewerBrain studies is: given one reviewer's history, how well
can we retrieve the historical reviews that matter for a new diff, and
eventually generate the review that reviewer would have written?

## Core Idea

Each reviewer's merged-PR history is mined into `(diff hunk → review
comment)` pairs. The pairs are cleaned, represented in a comment-first
embedding format, and indexed per reviewer in a local vector database.
Given a new diff, the system retrieves the most similar historical reviews
from that reviewer — reusable reviewer-specific knowledge that later
conditions a fine-tuned local model.

## Current Pipeline

```
GitHub history → preprocessing → reviewer-specific knowledge (RAG index)
              → retrieval → local LLM (planned) → personalized review
```

## Current Status

- **Data collection: complete** for the current experiment
  (thockin/kubernetes, ezyang/pytorch — see `data/raw/`).
- **Preprocessing: complete** — 1,995 thockin + 669 ezyang cleaned examples
  (2,664 combined), produced by `scripts/data/clean_dataset.py` plus
  `scripts/data/apply_clean_amendments.py`.
- **RAG: validated** — comment-first representation, ChromaDB cosine
  indexes, deterministic leakage-free held-out evaluation (496 queries).
- **Quantum-inspired retrieval (fidelity = cos²): evaluated** — produced
  *no measurable advantage* over cosine; cosine is the frozen metric.
- **Local LLM baseline + RAG inference: piloted** — Ollama +
  `qwen2.5-coder:7b` (Q4_K_M, RTX 4050 6 GB, greedy decoding), generic
  baseline vs RAG-conditioned prompts on 20 deterministic held-out queries
  (`evaluations/reports/local_llm_baseline.md`).
- **LoRA/QLoRA fine-tuning: NOT started.**

## Repository Structure

| Directory | Contents |
|---|---|
| `data/raw/` | Raw PR-level JSONL datasets (Git LFS) + reviewer-candidate CSV |
| `data/processed/` | Cleaned datasets (generated, gitignored, regenerable) |
| `src/reviewerbrain/` | Source package: representation, embeddings, split, metrics, config |
| `scripts/` | Runnable CLIs: `data/`, `retrieval/`, `evaluation/`, `inference/` |
| `configs/rag/default.yaml` | Frozen RAG configuration (mirrored by tests) |
| `configs/prompts/` | Versioned inference prompts (`baseline_v1`, `rag_v1`) |
| `indexes/chroma/v2/` | Validated ChromaDB indexes (generated, gitignored) |
| `evaluations/heldout/` | Frozen 496-query held-out evaluation artifact (LFS) |
| `evaluations/inference/` | LLM run artifacts: generations, manifests, metrics (generated, gitignored) |
| `evaluations/reports/` | Reports for the validated representation, evaluation, and local-LLM baseline |
| `docs/methodology/` | Reproducibility documentation |
| `docs/experiments/` | Historical stage reports (audit, cleaning, v1 RAG) |
| `experiments/archive/` | Superseded scripts (v1 RAG) |
| `tests/` | Unit tests for representation, split, paths, config |

Source vs generated: everything under `src/`, `scripts/`, `configs/`,
`tests/`, `docs/` is source; `data/processed/`, `indexes/`,
`evaluations/heldout/` are generated (see `docs/methodology/rag_reproduction.md`
for exact regeneration commands).

## Current RAG Configuration (frozen)

- Embedding model: `all-MiniLM-L6-v2`, 384 dims, 256-token window, CPU
- Document: comment-first — `Reviewer comment / File / Relevant diff`,
  diff anchored-trimmed to the commented line (exact rule in
  `src/reviewerbrain/retrieval/representation.py`)
- Index: ChromaDB, cosine space, per-reviewer collections
  (`indexes/chroma/v2/`)
- Retrieval: top-k = 3, cosine gate ≥ 0.5 (results below dropped)
- Evaluation split: per reviewer, every 5th PR (sorted by number) held out

## Evaluation

Held-out evaluation over 496 leakage-free queries (same-file hit@1/3/5 =
24.6/34.7/40.1%; same-directory 37.9/50.4/56.9%). Relevance is measured by
objective file-location proxies, not human labels — retrieval validation
only. **Reviewer-style generation has NOT been validated**; whether the
system reproduces a reviewer's voice is a question for the (not yet
started) fine-tuning stage. Known limitations are listed in
`evaluations/reports/rag_evaluation.md`.

## Reproduction

```bash
python -m pip install -r requirements.txt
# then see docs/methodology/rag_reproduction.md for the exact,
# step-by-step reproduction commands (cleaning → indexes → evaluation)
```

Local-LLM review generation (requires a running Ollama server on
127.0.0.1:11434 and the `qwen2.5-coder:7b` model):

```bash
python scripts/inference/run_review_generation.py --tag <run> --queries 10 --mode both
python scripts/inference/evaluate_generations.py \
    --run-dir evaluations/inference/<run> [--bertscore]
```

## Future Work

- Optional longer-context embedding experiment (MiniLM's 256-token window
  is the main representation constraint)
- LoRA/QLoRA reviewer adaptation on the cleaned datasets
- Local Ollama inference wired to the per-reviewer RAG indexes
- End-to-end evaluation of generated reviews against held-out history
