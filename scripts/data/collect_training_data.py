#!/usr/bin/env python3
"""
collect_training_data.py — ReviewerBrain training-data collector.

Extracts (code diff, reviewer comment) pairs for ONE chosen reviewer,
filtering to code-substantive comments, and detecting whether each
comment led to a code change in the same PR.

Usage:
    python collect_training_data.py \\
        --repo pandas-dev/pandas \\
        --reviewer jbrockmendel \\
        --lookback 500 \\
        --output jbrockmendel_training.jsonl \\
        [--dry-run] [--max-prs 50]

Requires:
    GITHUB_TOKEN environment variable or .env file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_training_data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

load_dotenv()

# Force UTF-8 output on Windows (cp1252 can't handle emoji in PR bodies)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

REST_BASE = "https://api.github.com"
RATE_LIMIT_BUFFER = 100
MAX_RETRIES = 10


# ═══════════════════════════════════════════════════════════════════════════
# GitHub REST Client  (shared with find_reviewer.py — kept inline to
# preserve the "single runnable script" constraint)
# ═══════════════════════════════════════════════════════════════════════════

class GitHubClient:
    """REST client with rate-limit backoff, retry, and throttle."""

    def __init__(self, token: str, delay: float = 0.3) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self.delay = delay
        self._last_call: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_call = time.monotonic()

    def _check_rate_limit(self, resp: requests.Response) -> None:
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset_at = resp.headers.get("X-RateLimit-Reset")
        if remaining is not None and int(remaining) < RATE_LIMIT_BUFFER and reset_at:
            sleep_sec = max(0, int(reset_at) - int(time.time())) + 5
            log.warning(
                "Rate-limit low (remaining=%s). Sleeping %ds until reset.",
                remaining, sleep_sec,
            )
            time.sleep(sleep_sec)

    def _backoff(self, resp: requests.Response, attempt: int) -> None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            sleep_sec = int(retry_after) + 1
        else:
            sleep_sec = min(10 * (2 ** attempt), 300)
        log.warning(
            "HTTP %d - attempt %d/%d, sleeping %ds.",
            resp.status_code, attempt + 1, MAX_RETRIES, sleep_sec,
        )
        time.sleep(sleep_sec)

    def get(self, path: str, params: dict | None = None) -> list | dict:
        url = f"{REST_BASE}{path}"
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self._session.get(url, params=params)
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.Timeout) as exc:
                if attempt < MAX_RETRIES - 1:
                    sleep_sec = min(10 * (2 ** attempt), 300)
                    log.warning(
                        "Network error (%s) - attempt %d/%d, sleeping %ds.",
                        type(exc).__name__, attempt + 1, MAX_RETRIES, sleep_sec,
                    )
                    time.sleep(sleep_sec)
                    continue
                raise
            self._check_rate_limit(resp)

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    self._backoff(resp, attempt)
                    continue
            resp.raise_for_status()
        raise RuntimeError(f"Exhausted retries for GET {url}")

    def get_paginated(
        self, path: str, per_page: int = 100, max_items: int | None = None
    ) -> list:
        items: list = []
        page = 1
        while True:
            data = self.get(path, params={"per_page": per_page, "page": page})
            if not data:
                break
            items.extend(data)
            if max_items and len(items) >= max_items:
                items = items[:max_items]
                break
            if len(data) < per_page:
                break
            page += 1
        return items


# ═══════════════════════════════════════════════════════════════════════════
# SQLite Cache — reuses the same schema/DB as find_reviewer.py so data
# fetched during the ranking step is never re-fetched.
# ═══════════════════════════════════════════════════════════════════════════

class PRCache:
    """Disk-backed cache keyed by (repo, pr_number).

    Schema matches find_reviewer.py so both scripts share the cache.
    This script extends the cache with extra tables for files and commits
    that the ranking script didn't need.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        # Original table from find_reviewer.py
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pr_data (
                pr_number  INTEGER PRIMARY KEY,
                repo       TEXT NOT NULL,
                pr_json    TEXT NOT NULL,
                reviews    TEXT NOT NULL,
                comments   TEXT NOT NULL
            )
        """)
        # New table: file patches per PR
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pr_files (
                pr_number  INTEGER,
                repo       TEXT NOT NULL,
                files_json TEXT NOT NULL,
                PRIMARY KEY (repo, pr_number)
            )
        """)
        # New table: commits per PR (for led_to_code_change detection)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pr_commits (
                pr_number    INTEGER,
                repo         TEXT NOT NULL,
                commits_json TEXT NOT NULL,
                PRIMARY KEY (repo, pr_number)
            )
        """)
        self._conn.commit()

    # ---- pr_data (reviews + comments) ----

    def has_pr(self, repo: str, pr_number: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM pr_data WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchone()
        return row is not None

    def get_pr(self, repo: str, pr_number: int) -> tuple[dict, list, list] | None:
        row = self._conn.execute(
            "SELECT pr_json, reviews, comments FROM pr_data "
            "WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), json.loads(row[1]), json.loads(row[2])

    def put_pr(
        self, repo: str, pr_number: int,
        pr_json: dict, reviews: list, comments: list,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO pr_data "
            "(pr_number, repo, pr_json, reviews, comments) VALUES (?,?,?,?,?)",
            (pr_number, repo,
             json.dumps(pr_json), json.dumps(reviews), json.dumps(comments)),
        )
        self._conn.commit()

    # ---- pr_files ----

    def has_files(self, repo: str, pr_number: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM pr_files WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchone()
        return row is not None

    def get_files(self, repo: str, pr_number: int) -> list | None:
        row = self._conn.execute(
            "SELECT files_json FROM pr_files WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put_files(self, repo: str, pr_number: int, files: list) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO pr_files "
            "(pr_number, repo, files_json) VALUES (?,?,?)",
            (pr_number, repo, json.dumps(files)),
        )
        self._conn.commit()

    # ---- pr_commits ----

    def has_commits(self, repo: str, pr_number: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM pr_commits WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchone()
        return row is not None

    def get_commits(self, repo: str, pr_number: int) -> list | None:
        row = self._conn.execute(
            "SELECT commits_json FROM pr_commits WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put_commits(self, repo: str, pr_number: int, commits: list) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO pr_commits "
            "(pr_number, repo, commits_json) VALUES (?,?,?)",
            (pr_number, repo, json.dumps(commits)),
        )
        self._conn.commit()

    def cached_pr_count(self, repo: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM pr_data WHERE repo = ?", (repo,),
        ).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# Comment filter — rule-based, swappable
# ═══════════════════════════════════════════════════════════════════════════

# Patterns that indicate a NON-code comment (process / logistics / acks).
_NON_CODE_PATTERNS: list[re.Pattern] = [
    # Pure approvals / acks
    re.compile(r"^(\+1|lgtm|looks\s+good(\s+to\s+me)?|ship\s+it)[\s.!]*$", re.I),
    # Thanks / greetings
    re.compile(r"^(thanks?(\s+you)?|ty|nice(\s+work)?|great(\s+work)?|awesome)[\s.!]*$", re.I),
    # Merge / rebase process
    re.compile(r"^(merg(ing|ed)(\s+now)?|can\s+you\s+rebase|please\s+rebase)[\s.!?]*$", re.I),
    # Pings
    re.compile(r"^(ping|cc)\s+@", re.I),
    # Whatsnew / changelog logistics (when that's the ENTIRE comment)
    re.compile(r"^please\s+add\s+(a\s+)?whatsnew[\s.]*$", re.I),
    # CI bot noise
    re.compile(r"^(ci|tests?|build)\s+(pass(ed|ing)?|fail(ed|ing)?|green|red)[\s.!]*$", re.I),
]

# Positive signals that a comment IS about code, even if short.
_CODE_SIGNAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"`[^`]+`"),                     # inline code
    re.compile(r"```"),                          # code block
    re.compile(r"(def|class|import|return)\s"),  # Python keywords
    re.compile(r"(should|could|might|can)\s+(be|use|return|raise|handle)", re.I),
    re.compile(r"(edge\s*case|off[- ]by[- ]one|race\s*condition)", re.I),
    re.compile(r"(type|typing|annotation|signature)", re.I),
    re.compile(r"(performance|allocat|memory|slow|fast|O\()", re.I),
    re.compile(r"(test|assert|fixture|parametri[zs]e)", re.I),
    re.compile(r"(deprecat|backward|compat|break)", re.I),
    re.compile(r"(naming|rename|variable|argument|param)", re.I),
    re.compile(r"(instead|rather|prefer|suggest|recommend|consider)", re.I),
    re.compile(r"(bug|fix|error|exception|raise|catch|handle)", re.I),
    re.compile(r"(refactor|simplif|extract|inline|DRY)", re.I),
    re.compile(r"(API|interface|public|private|internal)", re.I),
    re.compile(r"(nit|nitpick|style|format|pep)", re.I),
    re.compile(r"\bwhy\b.*\?", re.I),           # asking "why" → reasoning
    re.compile(r"this\s+(will|would|could|might)\s+(break|fail|crash)", re.I),
]


def is_code_related_comment(body: str, path: str | None, line: int | None) -> bool:
    """Rule-based filter: return True if the comment is about code.

    This function is designed to be **swappable** — replace it with an
    LLM-based classifier later without touching the rest of the pipeline.
    Signature contract: (body, path, line) -> bool.
    """
    text = (body or "").strip()

    # Empty comments are not useful
    if len(text) < 2:
        return False

    # Check against non-code patterns (exact full-comment matches)
    for pat in _NON_CODE_PATTERNS:
        if pat.match(text):
            return False

    # If it's attached to a specific file + line, it's almost certainly
    # about code — unless caught by the non-code patterns above.
    if path and line is not None:
        return True

    # If attached to a file path (even without a line), lean towards code.
    if path:
        # Still check if the body itself is substantive
        if len(text) >= 15:
            return True
        # Short comment on a file — check for code signals
        for pat in _CODE_SIGNAL_PATTERNS:
            if pat.search(text):
                return True
        return False

    # PR-level comment (no path, no line) — needs positive signal
    if len(text) < 10:
        return False

    for pat in _CODE_SIGNAL_PATTERNS:
        if pat.search(text):
            return True

    # Longer comments without explicit signals — keep if > 50 chars
    # (likely substantive discussion)
    if len(text) >= 50:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# "Led to code change" detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_code_change(
    comment_path: str | None,
    comment_line: int | None,
    comment_created_at: str,
    commits: list[dict],
    client: GitHubClient,
    owner: str,
    repo: str,
    cache: PRCache,
    pr_number: int,
) -> tuple[bool, str | None]:
    """Check if a comment led to a code change in the same PR.

    Looks for commits pushed AFTER the comment's created_at that touch
    the same file.  Returns (led_to_change, follow_up_patch_or_None).
    """
    if not comment_path:
        return False, None

    # Parse comment timestamp
    try:
        comment_ts = datetime.fromisoformat(
            comment_created_at.replace("Z", "+00:00")
        )
    except (ValueError, TypeError):
        return False, None

    # Find commits after the comment that might contain the response
    for commit in commits:
        commit_date_str = (
            commit.get("commit", {}).get("committer", {}).get("date", "")
            or commit.get("commit", {}).get("author", {}).get("date", "")
        )
        if not commit_date_str:
            continue
        try:
            commit_ts = datetime.fromisoformat(
                commit_date_str.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            continue

        if commit_ts <= comment_ts:
            continue

        # This commit was pushed after the comment — check if it touches
        # the same file.  We fetch the commit detail to get per-file patches.
        sha = commit.get("sha", "")
        if not sha:
            continue

        commit_detail = client.get(f"/repos/{owner}/{repo}/commits/{sha}")
        files_in_commit = commit_detail.get("files", [])

        for f in files_in_commit:
            if f.get("filename") == comment_path:
                return True, f.get("patch", "")

    return False, None


# ═══════════════════════════════════════════════════════════════════════════
# JSONL writer with resumability
# ═══════════════════════════════════════════════════════════════════════════

class JsonlWriter:
    """Append-mode JSONL writer. Loads seen PR numbers on init for resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.seen_prs: set[int] = set()
        self._fh = None
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        log.info("Scanning existing output for already-collected PRs ...")
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pr_num = obj.get("pr_number")
                    if pr_num is not None:
                        self.seen_prs.add(int(pr_num))
                except json.JSONDecodeError:
                    pass
        if self.seen_prs:
            log.info("Found %d already-written PRs - will skip.", len(self.seen_prs))

    def open(self) -> None:
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        if self._fh is None:
            raise RuntimeError("Writer not opened.")
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


# ═══════════════════════════════════════════════════════════════════════════
# Bot detection
# ═══════════════════════════════════════════════════════════════════════════

def is_bot(user: dict | None) -> bool:
    if user is None:
        return True
    if user.get("type", "").lower() == "bot":
        return True
    login = user.get("login", "").lower()
    return "bot" in login or login.endswith("[bot]")


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def list_merged_prs(
    client: GitHubClient,
    owner: str,
    repo: str,
    lookback: int,
) -> list[dict]:
    """List the last `lookback` merged PRs via REST (state=closed, filter merged)."""
    pr_list: list[dict] = []
    page = 1
    while len(pr_list) < lookback:
        data = client.get(
            f"/repos/{owner}/{repo}/pulls",
            params={
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": page,
            },
        )
        if not data:
            break
        for pr in data:
            if pr.get("merged_at"):
                pr_list.append(pr)
                if len(pr_list) >= lookback:
                    break
        if len(data) < 100:
            break
        page += 1
    return pr_list


def search_reviewed_prs(
    client: GitHubClient,
    owner: str,
    repo: str,
    reviewer: str,
    lookback: int,
) -> list[dict]:
    """Use GitHub Search API to find merged PRs reviewed by the target user.

    This is much more efficient for very active repos (e.g. pytorch/pytorch)
    where scanning chronologically would require tens of thousands of PRs
    to find a few dozen from a specific reviewer.

    GitHub Search API returns max 1000 results. We fetch the most recent
    `lookback` merged PRs that the reviewer commented on or reviewed.
    """
    pr_list: list[dict] = []
    page = 1
    # Search for merged PRs where the reviewer is involved as a commenter or reviewer
    # We use multiple search qualifiers to maximize coverage
    queries = [
        f"repo:{owner}/{repo} is:pr is:merged reviewed-by:{reviewer}",
        f"repo:{owner}/{repo} is:pr is:merged commenter:{reviewer}",
    ]

    seen_numbers: set[int] = set()

    for query in queries:
        page = 1
        while len(pr_list) < lookback:
            data = client.get(
                "/search/issues",
                params={
                    "q": query,
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            items = data.get("items", [])
            if not items:
                break
            for item in items:
                pr_num = item.get("number")
                if pr_num and pr_num not in seen_numbers:
                    seen_numbers.add(pr_num)
                    pr_list.append(item)
            if len(items) < 100:
                break
            page += 1
            # Search API caps at 10 pages (1000 results)
            if page > 10:
                break

    # Sort by updated_at descending, take top lookback
    pr_list.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
    return pr_list[:lookback]



def fetch_pr_review_data(
    client: GitHubClient,
    cache: PRCache,
    owner: str,
    repo: str,
    pr: dict,
    idx: int,
    total: int,
) -> tuple[dict, list, list]:
    """Fetch reviews + comments for one PR, using cache."""
    repo_full = f"{owner}/{repo}"
    pr_num = pr["number"]

    cached = cache.get_pr(repo_full, pr_num)
    if cached is not None:
        return cached

    log.info("  [%d/%d] PR #%d - fetching reviews + comments ...", idx, total, pr_num)

    # If the pr dict came from /search/issues, it won't have PR-specific
    # fields like merged_at at top level. Fetch full PR data via REST.
    if "merged_at" not in pr or pr.get("merged_at") is None:
        pr = client.get(f"/repos/{owner}/{repo}/pulls/{pr_num}")

    reviews = client.get_paginated(
        f"/repos/{owner}/{repo}/pulls/{pr_num}/reviews", per_page=100,
    )
    comments = client.get_paginated(
        f"/repos/{owner}/{repo}/pulls/{pr_num}/comments", per_page=100,
    )
    cache.put_pr(repo_full, pr_num, pr, reviews, comments)
    return pr, reviews, comments


def fetch_pr_files(
    client: GitHubClient,
    cache: PRCache,
    owner: str,
    repo: str,
    pr_number: int,
) -> list[dict]:
    """Fetch file patches for a PR, using cache."""
    repo_full = f"{owner}/{repo}"
    cached = cache.get_files(repo_full, pr_number)
    if cached is not None:
        return cached

    log.info("    Fetching file patches for PR #%d ...", pr_number)
    files = client.get_paginated(
        f"/repos/{owner}/{repo}/pulls/{pr_number}/files", per_page=100,
    )
    cache.put_files(repo_full, pr_number, files)
    return files


def fetch_pr_commits(
    client: GitHubClient,
    cache: PRCache,
    owner: str,
    repo: str,
    pr_number: int,
) -> list[dict]:
    """Fetch commits for a PR, using cache."""
    repo_full = f"{owner}/{repo}"
    cached = cache.get_commits(repo_full, pr_number)
    if cached is not None:
        return cached

    log.info("    Fetching commits for PR #%d ...", pr_number)
    commits = client.get_paginated(
        f"/repos/{owner}/{repo}/pulls/{pr_number}/commits", per_page=100,
    )
    cache.put_commits(repo_full, pr_number, commits)
    return commits


def reviewer_participated(
    reviews: list, comments: list, reviewer: str,
) -> bool:
    """Check if the target reviewer left any review or comment."""
    rl = reviewer.lower()
    for r in reviews:
        if not is_bot(r.get("user")):
            if (r.get("user") or {}).get("login", "").lower() == rl:
                return True
    for c in comments:
        if not is_bot(c.get("user")):
            if (c.get("user") or {}).get("login", "").lower() == rl:
                return True
    return False


def build_pr_record(
    pr: dict,
    reviews: list,
    comments: list,
    files: list,
    commits: list,
    reviewer: str,
    client: GitHubClient,
    owner: str,
    repo: str,
    cache: PRCache,
) -> tuple[dict, int, int, int]:
    """Build one JSONL record for a PR.

    Returns: (record, n_kept_comments, n_filtered_comments, n_led_to_change)
    """
    reviewer_lower = reviewer.lower()

    # -- files_changed --
    files_changed = []
    for f in files:
        files_changed.append({
            "filename":  f.get("filename", ""),
            "status":    f.get("status", ""),
            "patch":     f.get("patch", ""),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
        })

    # -- review_comments (inline, from target reviewer) --
    review_comments = []
    n_kept = 0
    n_filtered = 0
    n_led_to_change = 0

    for c in comments:
        user = c.get("user")
        if is_bot(user):
            continue
        if (user or {}).get("login", "").lower() != reviewer_lower:
            continue

        body = (c.get("body") or "").strip()
        path = c.get("path")
        line = c.get("original_line") or c.get("line")
        diff_hunk = c.get("diff_hunk", "")
        created_at = c.get("created_at", "")
        in_reply_to = c.get("in_reply_to_id")

        code_related = is_code_related_comment(body, path, line)
        if code_related:
            n_kept += 1
        else:
            n_filtered += 1

        # Detect led_to_code_change
        led_to_change, follow_up_patch = detect_code_change(
            comment_path=path,
            comment_line=line,
            comment_created_at=created_at,
            commits=commits,
            client=client,
            owner=owner,
            repo=repo,
            cache=cache,
            pr_number=pr["number"],
        )
        if led_to_change:
            n_led_to_change += 1

        review_comments.append({
            "comment_id":          c.get("id"),
            "path":                path,
            "line":                line,
            "diff_hunk":           diff_hunk,
            "body":                body,
            "created_at":          created_at,
            "in_reply_to_id":      in_reply_to,
            "is_code_related":     code_related,
            "led_to_code_change":  led_to_change,
            "follow_up_patch":     follow_up_patch,
        })

    # -- pr_level_reviews (from target reviewer) --
    pr_level_reviews = []
    for r in reviews:
        user = r.get("user")
        if is_bot(user):
            continue
        if (user or {}).get("login", "").lower() != reviewer_lower:
            continue
        state = r.get("state", "")
        body = (r.get("body") or "").strip()
        # Skip empty COMMENTED reviews (already captured as inline)
        if not body and state == "COMMENTED":
            continue
        pr_level_reviews.append({
            "state":        state,
            "body":         body,
            "submitted_at": r.get("submitted_at", ""),
        })

    record = {
        "pr_number":        pr["number"],
        "pr_title":         pr.get("title", ""),
        "pr_url":           pr.get("html_url", ""),
        "pr_description":   (pr.get("body") or ""),
        "author":           (pr.get("user") or {}).get("login", ""),
        "merged_at":        pr.get("merged_at", ""),
        "files_changed":    files_changed,
        "review_comments":  review_comments,
        "pr_level_reviews": pr_level_reviews,
    }

    return record, n_kept, n_filtered, n_led_to_change


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collect (code diff, reviewer comment) training data "
            "for one specific reviewer."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo", default="pandas-dev/pandas",
                   help='Repository as "owner/name".')
    p.add_argument("--reviewer", default="jbrockmendel",
                   help="GitHub username of the target reviewer.")
    p.add_argument("--lookback", type=int, default=500,
                   help="Number of most recent merged PRs to scan.")
    p.add_argument("--output", default=None,
                   help="Output JSONL file path.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print up to --max-prs matched PRs to stdout.")
    p.add_argument("--max-prs", type=int, default=None,
                   help="Stop after collecting this many matched PRs.")
    p.add_argument("--cache-dir", default=".reviewer_cache",
                   help="Directory for the SQLite cache DB.")
    p.add_argument("--delay", type=float, default=0.3,
                   help="Min seconds between API calls.")
    p.add_argument("--search", action="store_true",
                   help="Use GitHub Search API to find PRs reviewed by the target user. "
                        "Much more efficient for very active repos (e.g. pytorch/pytorch).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    # -- Token --
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        log.error(
            "GITHUB_TOKEN is not set.\n"
            "  Option 1: Paste your token in the .env file.\n"
            "  Option 2: export GITHUB_TOKEN=ghp_...\n"
            "  Create a PAT at: https://github.com/settings/tokens"
        )
        sys.exit(1)

    parts = args.repo.strip("/").split("/")
    if len(parts) != 2:
        log.error("--repo must be 'owner/name'. Got: %s", args.repo)
        sys.exit(1)
    owner, repo_name = parts
    reviewer = args.reviewer.strip()

    # -- Output --
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"{reviewer}_{repo_name}_training.jsonl")

    # -- Cache (reuses find_reviewer.py's DB if present) --
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_db = cache_dir / f"{owner}_{repo_name}.sqlite"
    cache = PRCache(cache_db)
    cached_before = cache.cached_pr_count(f"{owner}/{repo_name}")
    if cached_before:
        log.info("Cache has %d PRs from prior runs.", cached_before)

    # -- Client --
    client = GitHubClient(token=token, delay=args.delay)

    # -- Writer --
    writer: JsonlWriter | None = None
    if not args.dry_run:
        writer = JsonlWriter(output_path)
        writer.open()

    # -- Stats --
    start = time.monotonic()
    prs_scanned = 0
    prs_matched = 0
    prs_skipped_resume = 0
    total_comments_kept = 0
    total_comments_filtered = 0
    total_led_to_change = 0

    log.info(
        "Starting data collection: repo=%s/%s  reviewer=%s  "
        "lookback=%d  dry_run=%s  max_prs=%s",
        owner, repo_name, reviewer, args.lookback,
        args.dry_run, args.max_prs,
    )

    try:
        # Step 1: List merged PRs
        if args.search:
            log.info("Searching for up to %d merged PRs reviewed by '%s' ...", args.lookback, reviewer)
            pr_list = search_reviewed_prs(client, owner, repo_name, reviewer, args.lookback)
            log.info("Found %d merged PRs via search.", len(pr_list))
        else:
            log.info("Listing last %d merged PRs ...", args.lookback)
            pr_list = list_merged_prs(client, owner, repo_name, args.lookback)
            log.info("Found %d merged PRs.", len(pr_list))

        for i, pr_summary in enumerate(pr_list, 1):
            pr_num = pr_summary["number"]
            prs_scanned += 1

            # Resumability
            if writer and pr_num in writer.seen_prs:
                prs_skipped_resume += 1
                continue

            # Step 2: Fetch reviews + comments (cached)
            pr, reviews, comments = fetch_pr_review_data(
                client, cache, owner, repo_name, pr_summary, i, len(pr_list),
            )

            # Step 3: Filter to reviewer
            if not reviewer_participated(reviews, comments, reviewer):
                continue

            prs_matched += 1
            log.info(
                "PR #%d - reviewer '%s' participated (%d matched so far)",
                pr_num, reviewer, prs_matched,
            )

            # Step 4: Fetch file patches + commits (cached)
            files = fetch_pr_files(client, cache, owner, repo_name, pr_num)
            commits = fetch_pr_commits(client, cache, owner, repo_name, pr_num)

            # Step 5: Build record
            record, n_kept, n_filtered, n_change = build_pr_record(
                pr, reviews, comments, files, commits,
                reviewer, client, owner, repo_name, cache,
            )

            total_comments_kept += n_kept
            total_comments_filtered += n_filtered
            total_led_to_change += n_change

            log.info(
                "  -> %d comments kept, %d filtered, %d led to code change",
                n_kept, n_filtered, n_change,
            )

            if args.dry_run:
                print(json.dumps(record, ensure_ascii=False, indent=2))
            elif writer:
                writer.write(record)

            if args.max_prs and prs_matched >= args.max_prs:
                log.info("Reached --max-prs=%d. Stopping.", args.max_prs)
                break

    except KeyboardInterrupt:
        log.warning("Interrupted. Progress saved.")
    finally:
        if writer:
            writer.close()
        cache.close()

    # -- Summary --
    elapsed = time.monotonic() - start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)

    summary = [
        "",
        "=" * 56,
        "  Collection Complete",
        "=" * 56,
        f"  PRs scanned:                {prs_scanned:>8,}",
        f"  PRs skipped (resume):       {prs_skipped_resume:>8,}",
        f"  PRs with reviewer activity: {prs_matched:>8,}",
        f"  Comments kept (code):       {total_comments_kept:>8,}",
        f"  Comments filtered (non-code):{total_comments_filtered:>7,}",
        f"  Comments -> code change:    {total_led_to_change:>8,}",
        f"  Runtime:                    {h:02d}:{m:02d}:{s:02d}",
    ]
    if not args.dry_run:
        summary.append(f"  Output:                     {output_path}")
    summary.append("=" * 56)
    for line in summary:
        log.info(line)


if __name__ == "__main__":
    main()
