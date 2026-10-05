# ReviewerBrain

ReviewerBrain replicates the review behavior of a specific GitHub code
reviewer — not a generic reviewer — by converting that person's historical
PR review comments into a reviewer-specific retrieval knowledge base (RAG)
and fine-tuning reviewer-specific LoRA adapters on the same data. All
development runs locally with no cloud APIs and no data egress; LoRA
training is the one deliberate exception — it runs as a self-contained
workflow on Kaggle GPU.

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
              → retrieval ─┐
reviewer history → SFT → reviewer-specific LoRA (Kaggle, pending)
                           ▼
              four-mode generation (base / base_rag / lora / lora_rag)
              → personalized review → demo UI
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
- **SFT data pipeline + LoRA/QLoRA training implementation: complete** —
  leakage-guarded reviewer-specific SFT datasets, validated QLoRA
  configuration, Kaggle GPU training workflow, adapter registry, and the
  four-mode pipeline (`base` / `base_rag` / `lora` / `lora_rag`) with a
  demonstration UI. See `docs/methodology/lora_training.md`.
- **LoRA/QLoRA training: NOT yet performed** — no adapter, checkpoint, or
  training result exists. The Kaggle workflow (`kaggle/`) is ready; the
  expensive step runs there, not locally.
- **Four-mode generation evaluation: NOT started** (planned after the
  first Kaggle training run).

## Repository Structure

| Directory | Contents |
|---|---|
| `data/raw/` | Raw PR-level JSONL datasets (Git LFS) + reviewer-candidate CSV |
| `data/processed/` | Cleaned + SFT datasets (generated, gitignored, regenerable) |
| `src/reviewerbrain/` | Source package: representation, embeddings, split, metrics, config, training (SFT + config), inference (modes, adapters, backends) |
| `scripts/` | Runnable CLIs: `data/`, `retrieval/`, `evaluation/`, `inference/`, `training/` |
| `configs/rag/default.yaml` | Frozen RAG configuration (mirrored by tests) |
| `configs/prompts/` | Versioned inference/SFT prompts (`baseline_v1`, `rag_v1`, `sft_v1`) |
| `configs/training/` | LoRA/QLoRA training configuration (validated by tests) |
| `kaggle/` | Kaggle GPU training notebook + workflow guide |
| `ui/` | Streamlit demonstration UI (mock/loopback backends) |
| `adapters/`, `checkpoints/` | LoRA adapters + training checkpoints (gitignored — trained on Kaggle) |
| `indexes/chroma/v2/` | Validated ChromaDB indexes (generated, gitignored) |
| `evaluations/heldout/` | Frozen 496-query held-out evaluation artifact (LFS) |
| `evaluations/inference/` | LLM run artifacts: generations, manifests, metrics (generated, gitignored) |
| `evaluations/reports/` | Reports for the validated representation, evaluation, and local-LLM baseline |
| `docs/methodology/` | Reproducibility documentation (RAG + LoRA training) |
| `docs/experiments/` | Historical stage reports (audit, cleaning, v1 RAG) |
| `experiments/archive/` | Superseded scripts (v1 RAG) |
| `tests/` | Unit tests: representation, split, paths, config, SFT data, training, pipeline |

Source vs generated: everything under `src/`, `scripts/`, `configs/`,
`tests/`, `docs/`, `kaggle/`, `ui/` is source; `data/processed/`,
`indexes/`, `evaluations/inference/`, `adapters/`, `checkpoints/` are
generated (see `docs/methodology/rag_reproduction.md` and
`docs/methodology/lora_training.md` for exact regeneration commands).

## Current RAG Configuration (frozen)

- Embedding model: `all-MiniLM-L6-v2`, 384 dims, 256-token window, CPU
- Document: comment-first — `Reviewer comment / File / Relevant diff`,
  diff anchored-trimmed to the commented line (exact rule in
  `src/reviewerbrain/retrieval/representation.py`)
- Index: ChromaDB, cosine space, per-reviewer collections
  (`indexes/chroma/v2/`)
- Retrieval: top-k = 3, cosine gate ≥ 0.5 (results below dropped)
- Evaluation split: per reviewer, every 5th PR (sorted by number) held out

## LoRA fine-tuning stage (implemented — training pending)

Four deliberately separate generation modes
(`reviewerbrain.inference.pipeline`):

| mode | prompt | retrieval | adapter |
|---|---|---|---|
| `base` | `baseline_v1` | — | — |
| `base_rag` | `rag_v1` | frozen RAG (top-3, gate ≥ 0.5) | — |
| `lora` | `sft_v1` | — | reviewer-specific |
| `lora_rag` | `rag_v1` | frozen RAG | reviewer-specific |

- SFT data: PR-level leakage guards (frozen held-out PRs fully excluded,
  deterministic PR-level validation carve-out, `follow_up_patch` never
  rendered, RAG examples never used as training data). Built with
  `scripts/training/build_sft_dataset.py` → `data/processed/sft/`.
- Training: QLoRA on `Qwen2.5-Coder-7B-Instruct`, one unmerged adapter
  per reviewer, configured entirely by
  `configs/training/lora_qwen25coder7b.yaml`; executed on Kaggle GPU via
  `kaggle/train_lora.ipynb` (see `kaggle/README.md`). The local machine
  never loads a model.
- Status: implemented and unit-tested; **no adapter or training result
  exists yet** — `docs/methodology/lora_training.md` is the methodology
  and run log.

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

Local-LLM review generation (requires a running Ollama server on
127.0.0.1:11434 and the `qwen2.5-coder:7b` model):

```bash
python scripts/inference/run_review_generation.py --tag <run> --queries 10 --mode both
python scripts/inference/evaluate_generations.py \
    --run-dir evaluations/inference/<run> [--bertscore]
```

Four-mode pipeline (mock backend by default — no model loading; add
`--backend ollama` for a local server, adapters after a Kaggle run):

```bash
python scripts/training/build_sft_dataset.py          # SFT data (deterministic)
python scripts/inference/run_pipeline.py --tag <run> --mode all
streamlit run ui/app.py                               # demonstration UI
```

## Future Work

- Run the Kaggle QLoRA training (`kaggle/train_lora.ipynb`) and log the
  run in `docs/methodology/lora_training.md`
- Four-mode generation evaluation over the held-out queries
  (`scripts/inference/run_pipeline.py`) — including whether LoRA closes
  the reviewer-concern gap RAG alone did not
- Optional longer-context embedding experiment (MiniLM's 256-token window
  is the main representation constraint)
- Human evaluation of generated reviews
