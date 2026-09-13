# Data

## Raw (`data/raw/`)

PR-level JSONL datasets mined from GitHub with `scripts/data/collect_training_data.py`
(requires `GITHUB_TOKEN` in the environment or a `.env` file). One JSONL
line = one pull request; review comments are nested inside each PR record.

Top-level PR fields: `pr_number`, `pr_title`, `pr_url`, `pr_description`,
`author` (the PR author, NOT the reviewer), `merged_at`, `files_changed`,
`review_comments[]` (the training signal), `pr_level_reviews[]`.

Each item in `review_comments[]`: `comment_id`, `path`, `line`,
`diff_hunk`, `body`, `created_at`, `in_reply_to_id`, `is_code_related`,
`led_to_code_change`, `follow_up_patch`.

| File | Reviewer | Repository | Size |
|---|---|---|---|
| `thockin_kubernetes_training.jsonl` | thockin (Tim Hockin) | kubernetes/kubernetes | ~95 MB, 839 PRs |
| `ezyang_pytorch_training.jsonl` | ezyang (Edward Z. Yang) | pytorch/pytorch | ~20 MB, 499 PRs |

Both files are tracked via **Git LFS** (`.gitattributes`: `*.jsonl`).

`liggitt_kubernetes_training.jsonl` and `pohly_kubernetes_training.jsonl`
are additional mined datasets kept locally for possible future experiments
but **not tracked in Git** (out of scope for the current pipeline).

`kubernetes_reviewers.csv` is the reviewer-candidate ranking produced by
`scripts/data/find_reviewer.py` that led to the thockin selection.

## Processed (`data/processed/`) — generated

Produced from `data/raw/` by `scripts/data/clean_dataset.py` followed by
`scripts/data/apply_clean_amendments.py` (exact commands in
`docs/methodology/rag_reproduction.md`). Gitignored because they are
deterministically regenerable from the raw data.

| File | Examples | Notes |
|---|---|---|
| `thockin_clean.jsonl` | 1,995 | after cleaning + approved amendments |
| `ezyang_clean.jsonl` | 669 | after cleaning + approved amendments |
| `combined_clean.jsonl` | 2,664 | concatenation of both (thockin first) |

Cleaned-record schema (exactly 11 fields — see
`src/reviewerbrain/data/schema.py`): `reviewer`, `repo`, `pr_number`,
`pr_title`, `pr_description`, `file`, `diff_hunk`, `review_comment`,
`is_code_related`, `led_to_code_change`, `follow_up_patch`.

`follow_up_patch` is **metadata only** — it occurs after the review and is
never model input or indexed text.
