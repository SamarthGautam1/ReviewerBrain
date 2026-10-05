# LoRA/QLoRA Training — Methodology and Reproduction

Status: **implemented, NOT yet trained.** Every artifact this document
describes exists as code and configuration; no adapter, checkpoint, loss
value, or generation result exists yet. When a Kaggle run completes,
record what actually happened under "Run log" below — never pre-fill.

## Objective

Given the reviewer identity and a code change, generate the kind of review
comment that reviewer would write. The supervised target is the cleaned
historical `review_comment`; the model is `Qwen2.5-Coder-7B-Instruct`
(HF instruct weights — the Ollama `qwen2.5-coder:7b` Q4_K_M artifact is a
serving quantization, never the training base).

One adapter per reviewer (`thockin`, `ezyang`), QLoRA (4-bit nf4 base,
fp16 LoRA on all attention + MLP projections, r=16, α=32, dropout 0.05),
adapters kept unmerged so the four inference modes stay separable.

## SFT data rules (frozen; enforced in code, tested in tests/test_sft_data.py)

Source: `data/processed/<reviewer>_clean.jsonl` (the approved cleaning
funnel; regenerated deterministically, see rag_reproduction.md).

1. **PR-level splitting only.** The frozen held-out rule
   (`reviewerbrain.evaluation.split`: per reviewer, every 5th sorted PR,
   position % 5 == 4) marks the protected evaluation PRs. NO record from a
   held-out PR enters train or validation — the identical guard the
   serving retrieval applies, so the SFT set, the RAG index candidates,
   and the frozen 496-query held-out artifact can never share a PR.
2. **Validation carve-out**, also PR-level and deterministic: from the
   remaining sorted PRs, every 10th (position % 10 == 9) becomes the
   training-time validation monitor. No RNG anywhere.
3. **`follow_up_patch` is never read** into any example text — it
   post-dates the review (future-information leakage). The builder
   re-verifies this against its own output before writing.
4. **Labels are not features**: `led_to_code_change` / `is_code_related`
   are never rendered into examples.
5. **RAG examples are inference-time evidence, not training data.** The
   LoRA training set contains no retrieved examples. (A separate
   "RAG-augmented SFT" experiment, if ever wanted, must be explicitly
   implemented, documented and named — it does not exist.)
6. **Target fidelity**: the assistant message is the `review_comment`
   verbatim; verified post-build on the exact objects written.

Actual counts (deterministic, reproducible):

| reviewer | records | held-out PRs (records) | train | val |
|---|---|---|---|---|
| thockin | 1,995 | 67 (358) | 1,443 | 194 |
| ezyang | 669 | 39 (138) | 490 | 41 |

## Prompt format (train/serve consistency)

Examples are rendered from the versioned template
`configs/prompts/sft_v1.yaml` — the same file `mode: lora` serves:

```
system: "You are writing the review comment that the GitHub reviewer
         {reviewer} would leave. …"
user:    "File: {file_path}\n{pr_context}\nCode change (diff hunk):
          ```diff\n{code_change}\n``` …"
assistant: the historical review_comment
```

The user content reuses the inference harness caps verbatim (diff ≤ 6,000
chars, description ≤ 1,000 chars, `heldout_queries`), so the training
distribution matches the served prompt distribution. The core task
instruction is identical to baseline_v1/rag_v1 so mode differences stay
attributable to the mechanism (identity / evidence / adaptation), not
wording.

Tokenization for training: Qwen chat template, assistant-only loss
(labels −100 everywhere except the assistant response + its closing
`<|im_end|>`); prefix-consistency is checked, over-long sequences are
left-truncated keeping the answer, and the truncation count is reported.

## Configuration

Single source: `configs/training/lora_qwen25coder7b.yaml`, loaded and
strictly validated by `reviewerbrain.training.config` (tests assert the
validation and the repo config's sanity; fp16 is used because Kaggle
T4/P100 GPUs have no bf16). Nothing is hardcoded in the training script.

## Execution (Kaggle only)

The local development machine never loads a model. The workflow —
environment, dependency install, data attachment, deterministic SFT
build, `--prepare-only` pre-flight, smoke run, real training, report
inspection, adapter packaging — is documented in `kaggle/README.md` and
implemented in `kaggle/train_lora.ipynb`.

Entry point: `scripts/training/train_lora.py` (heavy imports are lazy;
`--prepare-only` runs the full data path with zero model loading).

Outputs per reviewer: `adapters/<reviewer>/` with the unmerged PEFT
adapter, tokenizer files, and `training_report.json` (base model, config
+ SHA-256, seed, git commit, package versions, dataset stats, loss
history, tokenization flags). `adapters/` is gitignored — binaries stay
out of git; the report travels with the adapter.

## Four-mode evaluation structure

| mode | prompt | retrieval | adapter |
|---|---|---|---|
| base | baseline_v1 | — | — |
| base_rag | rag_v1 | frozen RAG | — |
| lora | sft_v1 | — | reviewer |
| lora_rag | rag_v1 | frozen RAG | reviewer |

Implemented in `reviewerbrain.inference.pipeline` +
`scripts/inference/run_pipeline.py` (records mirror the pilot harness, so
`evaluate_generations.py` works unchanged). Generation evaluation over
the held-out queries in all four modes is the planned post-training
experiment; it has NOT been run.

## Research framing

The validated finding stands: RAG changes the generated reviewer
register/style, but substantive reviewer-concern replication is
unresolved. The LoRA stage is the next experiment — nothing in this
repository may claim reviewer-replication results before adapters are
trained and evaluated.

## Run log

(no runs yet — append entries only after actual Kaggle runs)
