# ReviewerBrain — `collect_reviews.py`

A resumable Python script that mines GitHub pull-request history and extracts
`(code diff, reviewer comment)` pairs for **one specific reviewer**, formatted
as JSON Lines for LLM fine-tuning.

---

## Prerequisites

- Python 3.10+
- A GitHub Personal Access Token (PAT) with at least **`repo` scope** (or
  `public_repo` for public repositories only).

```
pip install -r requirements.txt
```

---

## Setting your GitHub token

**Linux / macOS / Git Bash:**
```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Windows PowerShell:**
```powershell
$env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Windows Command Prompt:**
```cmd
set GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ Never hard-code or commit your token. The script refuses to start if
> `GITHUB_TOKEN` is unset.

---

## Usage

### Dry run (no file written — prints 5 PRs to stdout)
```bash
python collect_reviews.py \
  --repo python/cpython \
  --reviewer gvanrossum \
  --dry-run \
  --max-prs 5
```

### Capped run (write up to 50 matched PRs)
```bash
python collect_reviews.py \
  --repo pandas-dev/pandas \
  --reviewer jreback \
  --output jreback_reviews.jsonl \
  --max-prs 50
```

### Full run (all merged PRs, resumable)
```bash
python collect_reviews.py \
  --repo pandas-dev/pandas \
  --reviewer jreback \
  --output jreback_reviews.jsonl
```

### Resume an interrupted run
Simply re-run the exact same command. The script reads the output file on
startup, notes which PR numbers are already present, and skips them.

```bash
# First run (killed after 10 minutes):
python collect_reviews.py --repo pandas-dev/pandas --reviewer jreback \
  --output jreback_reviews.jsonl

# Resume (picks up where it left off):
python collect_reviews.py --repo pandas-dev/pandas --reviewer jreback \
  --output jreback_reviews.jsonl
```

---

## All flags

| Flag | Default | Description |
|---|---|---|
| `--repo` | *(required)* | Repository as `owner/name` |
| `--reviewer` | *(required)* | GitHub username to filter by |
| `--output` | `{reviewer}_{repo}_reviews.jsonl` | Output file path |
| `--dry-run` | `False` | Print to stdout, don't write file |
| `--max-prs` | *(unlimited)* | Stop after N matched PRs |
| `--delay` | `0.5` | Seconds between API calls |
| `--page-size` | `50` | PRs per GraphQL page (reduce if node-limit errors occur) |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Output schema (JSON Lines)

One JSON object per line, one PR per object:

```json
{
  "pr_number": 12345,
  "pr_title": "Fix memory leak in read_csv",
  "pr_url": "https://github.com/pandas-dev/pandas/pull/12345",
  "pr_description": "This PR fixes ...",
  "author": "contributor_username",
  "merged_at": "2023-06-15T14:23:01Z",
  "files_changed": [
    {
      "filename": "pandas/io/parsers.py",
      "status": "modified",
      "patch": "@@ -100,7 +100,7 @@ ...",
      "additions": 3,
      "deletions": 1
    }
  ],
  "review_comments": [
    {
      "comment_id": 987654321,
      "path": "pandas/io/parsers.py",
      "line": 104,
      "diff_hunk": "@@ -100,7 +100,7 @@\n ...",
      "body": "This should use `_ensure_index` instead.",
      "created_at": "2023-06-14T09:11:00Z",
      "in_reply_to_id": null
    }
  ],
  "pr_level_reviews": [
    {
      "state": "CHANGES_REQUESTED",
      "body": "A few nits below — please address before merging.",
      "submitted_at": "2023-06-14T09:11:00Z"
    }
  ]
}
```

**`review_comments`** are inline comments attached to a specific file and line,
including the surrounding diff hunk for context.

**`pr_level_reviews`** are PR-level summaries (APPROVED / CHANGES_REQUESTED /
DISMISSED). Pure COMMENTED reviews with no body text are omitted (they are
already captured as inline comments).

---

## Rate limits & expected runtime

The script uses a **hybrid strategy**: GraphQL for cheap reviewer-participation
discovery (one paginated query covers ~50 PRs), then REST only for file patches
on matched PRs. This is dramatically more efficient than pure REST for large repos.

| Pool | Limit (authenticated) | Notes |
|---|---|---|
| REST | 5,000 req / hr | Used for `/pulls/{n}/files` only |
| GraphQL | 5,000 pts / hr | Used for PR discovery pages |

**Automatic safeguards built into the script:**
- Monitors `X-RateLimit-Remaining` after every call; sleeps to reset time if
  quota drops below 50 (overridable via `RATE_LIMIT_BUFFER` env var).
- Retries HTTP 403 / 429 / 5xx with exponential backoff (up to 5 attempts).
- Detects GraphQL-level `RATE_LIMITED` errors and sleeps 60 s.
- Configurable `--delay` (default 0.5 s) between calls to avoid GitHub's
  secondary (undocumented) rate limiter.

**Estimated runtime:**

| Repo size (merged PRs) | Reviewer match rate | Estimated runtime |
|---|---|---|
| 1,000 | 20 % | ~5 min |
| 5,000 | 10 % | ~25 min |
| 10,000 | 10 % | ~50 min |
| 50,000 | 5 % | ~4 hrs (spans ≥2 rate-limit windows) |

For very large repos, use `--max-prs` to run in batches overnight. Because the
script is resumable, you can chain multiple capped runs safely.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `GITHUB_TOKEN environment variable is not set` | Set the env var (see above) |
| `GraphQL node limit exceeded. Reduce --page-size.` | Add `--page-size 25` |
| `HTTP 401` | Token is invalid or expired — generate a new PAT |
| Output file has duplicate lines | Should never happen; if it does, deduplicate with `sort -u` or the Python one-liner below |

**Deduplicate output (if needed):**
```python
import json
from pathlib import Path

p = Path("jreback_reviews.jsonl")
seen = set()
lines = []
for line in p.read_text().splitlines():
    obj = json.loads(line)
    if obj["pr_number"] not in seen:
        seen.add(obj["pr_number"])
        lines.append(line)
p.write_text("\n".join(lines) + "\n")
```

**Validate schema after collection:**
```python
import json
required = {"pr_number","pr_title","pr_url","pr_description","author",
            "merged_at","files_changed","review_comments","pr_level_reviews"}
with open("jreback_reviews.jsonl") as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        missing = required - obj.keys()
        if missing:
            print(f"Line {i}: missing keys {missing}")
print("Validation complete.")
```
