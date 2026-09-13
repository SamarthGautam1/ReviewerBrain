# Architecture Overview

## Pipeline

```
GitHub PR history (raw JSONL, data/raw/, Git LFS)
        │  scripts/data/collect_training_data.py   (collection — complete)
        ▼
Cleaned per-example records (data/processed/, generated)
        │  scripts/data/clean_dataset.py + apply_clean_amendments.py
        ▼
Comment-first representation (src/reviewerbrain/retrieval/representation.py)
        │  scripts/retrieval/build_indexes.py
        ▼
Per-reviewer ChromaDB indexes (indexes/chroma/v2/, generated)
        │  cosine similarity, top-k=3, gate ≥ 0.5
        ▼
Retrieved historical reviews  ──(future: LoRA-tuned local LLM via Ollama)──▶  personalized review
```

## Source vs generated

| Source (committed) | Generated (gitignored, regenerable) |
|---|---|
| `src/reviewerbrain/` | `data/processed/*.jsonl` |
| `scripts/` | `indexes/chroma/v2/` |
| `configs/rag/default.yaml` | (reports under `evaluations/reports/` are |
| `tests/` | committed snapshots of generated runs) |
| `docs/`, `README.md` | |

The frozen evaluation artifact `evaluations/heldout/eval_heldout_queries.jsonl`
is generated but committed (Git LFS) as the regression baseline: a rerun of
`scripts/evaluation/run_heldout_eval.py` must reproduce it.

## Module map

- `reviewerbrain.paths` — the only place that resolves filesystem
  locations; keeps the repo portable.
- `reviewerbrain.retrieval.representation` — the comment-first document
  and query formats plus the anchored diff-trim rule. This is the core of
  the validated representation.
- `reviewerbrain.embeddings.model` — frozen embedding model constants and
  CPU loader.
- `reviewerbrain.evaluation.split` — deterministic PR-level held-out split.
- `reviewerbrain.evaluation.metrics` — Spearman correlation, boilerplate
  detection used in qualitative inspection.
- `reviewerbrain.data.schema` — the 11-field cleaned-record contract.
- `reviewerbrain.config` — loads `configs/rag/default.yaml` (tests keep
  config and code constants in sync).

## Design invariants

1. The review comment is always the first block of an indexed document and
   its tokens are a prefix of the embedding input (the 256-token model
   window can only truncate the diff tail).
2. `follow_up_patch` never enters a document, query, or training text — it
   post-dates the review and would leak future information.
3. The evaluation split is PR-level and deterministic; no query's PR may
   appear in the index.
4. Threshold comparisons between similarity metrics must be
   threshold-equivalent (cos ≥ 0.5 ⇔ fid = cos² ≥ 0.25).
