# Kaggle Training Workflow

ReviewerBrain trains reviewer-specific LoRA adapters on **Kaggle GPU**. The
local development machine never loads a model, never trains, and never
serves the 7B model — it only prepares code, configuration and data.

Status: **the workflow below is implemented but has NOT been run.** When a
run completes, update `docs/methodology/lora_training.md` with what
actually happened (date, hardware, config SHA, losses) — nothing may be
pre-filled.

## What trains

- Base model: `Qwen/Qwen2.5-Coder-7B-Instruct` (HF instruct weights — NOT
  the Ollama Q4_K_M serving artifact).
- Method: QLoRA — 4-bit nf4 quantized base, fp16 LoRA adapters on all
  attention + MLP projections (r=16, α=32, dropout 0.05), adapters kept
  separate from the base (never merged) so `mode: lora` / `mode: lora_rag`
  can load them.
- Data: the frozen SFT datasets rebuilt on Kaggle by
  `scripts/training/build_sft_dataset.py` (deterministic; same output as
  local). Leakage guards are inside the pipeline, not the notebook.
- One adapter per reviewer: `thockin`, `ezyang` (two runs of the same
  script; reviewers come from `configs/training/lora_qwen25coder7b.yaml`).

## Notebook

`kaggle/train_lora.ipynb` — cells in order:

1. **Environment check** — `nvidia-smi`, torch/CUDA versions. Use GPU
   **T4 x2** or **P100**; internet **ON**.
2. **Clone + install** — shallow clone of this repository and
   `pip install -r requirements-training.txt` (training-only deps: torch,
   transformers, accelerate, peft, bitsandbytes).
3. **Cleaned corpora** — copies `thockin_clean.jsonl` /
   `ezyang_clean.jsonl` from an attached Kaggle dataset into
   `data/processed/`. These are the deterministic output of the approved
   cleaning pipeline; attach them as a Kaggle dataset (they are gitignored
   build artifacts, so they are not in the repository itself).
4. **SFT build** — runs `build_sft_dataset.py`; the leakage guards run
   here on Kaggle exactly as locally; stats must match local output.
5. **`--prepare-only`** — full data path with zero model loading.
6. **Smoke run** — `--smoke` (16 examples) per reviewer to verify the
   whole path cheaply.
7. **Real training** — `scripts/training/train_lora.py --reviewer <r>`
   per reviewer.
8. **Reports + packaging** — prints the `training_report.json` summaries
   and zips `adapters/` into `/kaggle/working/reviewerbrain_adapters.zip`.

The only placeholder is `CLEANED_DATA_SOURCE` (the path of the Kaggle
dataset you attach). No credentials or private slugs are used.

## Outputs

`/kaggle/working/adapters/<reviewer>/` contains:

- `adapter_config.json` + `adapter_model.safetensors` — the PEFT LoRA
  adapter (unmerged, used by `mode: lora` / `mode: lora_rag`)
- tokenizer files
- `training_report.json` — base model, full config + SHA-256, seed, git
  commit, package versions, dataset stats, loss history. This file is the
  reproducibility record; keep it with the adapter.

## After training

1. Download the zip; unpack under `<repo>/adapters/<reviewer>/` on the
   serving machine. `adapters/` is gitignored (binaries stay out of git).
2. Record the run in `docs/methodology/lora_training.md` — actual values
   only.
3. Generation evaluation over the held-out queries in the four modes
   (`scripts/inference/run_pipeline.py`) is the follow-up experiment; it
   has not been run.

## Reproducibility notes

- Splitting, cleaning and prompt rendering are deterministic (no RNG).
- Training randomness is bounded by `seed: 42` in the config (torch /
  transformers seeds set by HF `Trainer`); GPU nondeterminism means
  loss curves may differ slightly across reruns — the report captures
  the actuals.
- The SFT prompt (`configs/prompts/sft_v1.yaml`) is versioned and shared
  between training and `mode: lora` serving, so they cannot drift.
