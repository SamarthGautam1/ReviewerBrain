# RAG Stage Reproduction

This document fully specifies the validated RAG stage. It assumes the
repository layout at the repo root; all commands run from the root on
Linux/macOS/Git Bash. Everything was validated on Python 3.13 with the
pinned versions in `requirements.txt`.

## 1. Input data

- `data/raw/thockin_kubernetes_training.jsonl` — 839 PRs,
  3,405 review comments (kubernetes/kubernetes, reviewer thockin)
- `data/raw/ezyang_pytorch_training.jsonl` — 499 PRs,
  1,244 review comments (pytorch/pytorch, reviewer ezyang)

One JSONL line = one PR; training signal = nested `review_comments[]`
(fields: `comment_id`, `path`, `line`, `diff_hunk`, `body`,
`in_reply_to_id`, `is_code_related`, `led_to_code_change`,
`follow_up_patch`). Raw files are tracked via Git LFS.

## 2. Preprocessing

`scripts/data/clean_dataset.py` flattens PR wrappers and applies, in order:

1. drop replies (`in_reply_to_id` not null)
2. drop `body` shorter than 5 chars
3. drop comments whose body is only one ```suggestion``` fenced block
4. keep source-code files only (exclude `.md .rst .txt .adoc .tex .yaml
   .yml .json .toml .ini .cfg .conf .xml .html .htm .csv .tsv .svg`;
   exclude extensionless docs by basename — README, LICENSE, OWNERS, ...;
   keep build/codegen files: `.sh`, `.cmake`, `.expect`, `.proto`,
   `.mod`, `.sum`, `.patch`, Makefile, Dockerfile, ...)
5. dedup on `(pr_number, path, body)`, keep first occurrence

`scripts/data/apply_clean_amendments.py` then applies two approved
amendments: drop examples with empty `diff_hunk` (7 thockin), drop the
ezyang `(test/dynamo/test_export.py.bak, "delete me")` example (1).

Funnel: thockin 3,405 → 2,002 → 1,995; ezyang 1,244 → 670 → 669.
Final: **1,995 + 669 = 2,664 examples** in `data/processed/`.

## 3. Document representation

Implemented in `src/reviewerbrain/retrieval/representation.py`
(`doc_text`). Comment-first so the review comment is always inside the
embedding window:

```
Reviewer comment:
<review_comment>

File:
<file path>

Relevant diff:
<trimmed diff context>
```

Exact truncation rule:

- window = 256 wordpiece tokens (`all-MiniLM-L6-v2` max_seq_length); 2
  reserved for `[CLS]`/`[SEP]`, 2 safety.
- diff token budget = window − comment − path − scaffold − 4, floored at
  24 tokens. The comment is never cut (it is first); if the comment alone
  exceeds the window the example is flagged
  `comment_overflows_window` (59 docs, 2.2%) and the model truncates only
  the diff tail.
- diff trimming: parse the `@@ -a,b +c,d @@` header, assign each body line
  its new-file line number, anchor at the line matching the comment's
  `line` field (nearest if out of range; midpoint of `+`/`-` lines when
  `line` is null), grow a contiguous window outward (nearest line first)
  until the budget is reached. No elision markers; the `@@` header is
  dropped; an oversized single line is hard-truncated proportionally.
- the comment's tokens are guaranteed to be an exact token-prefix of the
  encoded document (verified in `tests/test_representation.py` and by
  `scripts/retrieval/build_indexes.py` over all 2,664 docs).

`follow_up_patch` is never used in any representation.

## 4. Embedding

`all-MiniLM-L6-v2` (384 dimensions, max_seq_length 256), sentence-
transformers 6.0.1, CPU, float32, batch 64. Query embeddings are L2-
normalized for evaluation scoring.

## 5. Indexing

`scripts/retrieval/build_indexes.py` builds three ChromaDB persistent
collections under `indexes/chroma/v2/` (cosine space, telemetry disabled):

| collection | docs |
|---|---|
| `thockin` | 1,995 |
| `ezyang` | 669 |
| `combined` | 2,664 |

Document metadata: `reviewer`, `pr_number`, `pr_title`, `file`,
`comment_id`, `review_comment`, `led_to_code_change`.

## 6. Query construction

`representation.query_text` — same trim rule, no comment block:

```
File:
<file path>

Relevant diff:
<trimmed diff context>
```

Queries use the freed comment budget for diff context.

## 7. Retrieval (frozen serving configuration)

- per-reviewer indexes (`thockin`, `ezyang`) for serving
- top-k = 3
- cosine similarity gate ≥ 0.5 — results below are dropped
- the `combined` index exists but per-reviewer retrieval is the frozen
  serving configuration

## 8. Evaluation split

Deterministic, no RNG (`src/reviewerbrain/evaluation/split.py`): per
reviewer, sorted unique `pr_number`s, every 5th PR (position % 5 == 4)
held out. Queries = all held-out examples; the retrieval index contains
no example from any held-out PR (same-PR leakage impossible; verified 0
leaks). Query/index sizes: thockin 358 queries / 1,637 index docs;
ezyang 138 / 531. Total **496 queries**.

## 9. Metrics

`scripts/evaluation/run_heldout_eval.py` reports, for cosine and for
fidelity = cos² (equivalent gates: cos ≥ 0.5 vs fid ≥ 0.25, plus a
no-threshold run, over the same embeddings):

- same-file and same-directory hit@1/3/5 (objective relevance proxies —
  NOT human-labeled relevance)
- mean score of top-k
- gated hit@k and zero-candidate query counts
- cosine∩fidelity top-k overlap
- Spearman rank correlation over full candidate lists
- sign-fold occurrences (negative-cosine members of fidelity top-5)
- same-PR leak count (must be 0)

## 10. Reproducing the validated results

```bash
# 0) environment (Python 3.10+; validated on 3.13)
python -m pip install -r requirements.txt
python -m pip install pytest

# 1) raw data — already present via Git LFS (git lfs pull), or re-collect:
#    GITHUB_TOKEN=<token> python scripts/data/collect_training_data.py \
#        --repo kubernetes/kubernetes --reviewer thockin --search \
#        --output data/raw/thockin_kubernetes_training.jsonl

# 2) clean (scripts print sections; the shell splits them into files)
python scripts/data/clean_dataset.py > /tmp/clean_stream.txt
awk '/^===THOCKIN===/{f="data/processed/thockin_clean.jsonl";next}
     /^===EZYANG===/{f="data/processed/ezyang_clean.jsonl";next}
     /^===REPORT===/{f="evaluations/reports/cleaning_report.md";next}
     f{print > f}' /tmp/clean_stream.txt
cat data/processed/thockin_clean.jsonl data/processed/ezyang_clean.jsonl \
    > data/processed/combined_clean.jsonl
python scripts/data/apply_clean_amendments.py > /tmp/amend_stream.txt
awk '/^===THOCKIN===/{f="data/processed/thockin_clean.jsonl";next}
     /^===EZYANG===/{f="data/processed/ezyang_clean.jsonl";next}
     /^===SUMMARY===/{f="";next}
     f{print > f}' /tmp/amend_stream.txt
cat data/processed/thockin_clean.jsonl data/processed/ezyang_clean.jsonl \
    > data/processed/combined_clean.jsonl
# expected: 1995 / 669 / 2664 lines

# 3) build indexes (writes indexes/chroma/v2/, ~2 min on CPU)
python scripts/retrieval/build_indexes.py > /tmp/build_report.md

# 4) held-out evaluation (read-only over the indexes; ~1-2 min)
python scripts/evaluation/run_heldout_eval.py > /tmp/eval_stream.txt
awk '/^===QUERIES===/{f="evaluations/heldout/eval_heldout_queries.jsonl";next}
     /^===REPORT===/{f="evaluations/reports/rag_evaluation.md";next}
     f{print > f}' /tmp/eval_stream.txt

# 5) unit tests
python -m pytest tests/ -q
```

### Frozen reference numbers (must reproduce)

- cleaned counts: 1,995 / 669 / 2,664
- indexes: 1,995 / 669 / 2,664 docs, 384 dims
- evaluation: 496 queries (358 thockin + 138 ezyang); 0 same-PR leaks
- same-file hit@1/3/5 (no thresh): 24.6% / 34.7% / 40.1%
- same-dir hit@1/3/5: 37.9% / 50.4% / 56.9%
- mean top-1 cosine: 0.7023
- zero-candidate queries (gated): 9
- cosine∩fidelity top-k overlap: 1.00 / 3.00 / 5.00 (identical ranking;
  fidelity offers no advantage — cosine is frozen)
- the only value that may differ between runs is the reported query
  embedding time (a wall-clock measurement)

Note: `experiments/archive/step3_rag.py` is a historical artifact with
period-correct absolute paths; it is intentionally not runnable and not
covered by the stale-path checks.
