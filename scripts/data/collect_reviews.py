#!/usr/bin/env python3
"""
collect_reviews.py — ReviewerBrain data-collection script.

Mines a GitHub repository's pull-request history and extracts
(code diff, reviewer comment) pairs for ONE target reviewer.

Usage:
    python collect_reviews.py \\
        --repo pandas-dev/pandas \\
        --reviewer jreback \\
        --output jreback_reviews.jsonl \\
        [--dry-run] [--max-prs 50] [--delay 0.5] [--page-size 50]

Requires:
    GITHUB_TOKEN environment variable (Personal Access Token with repo scope).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

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
log = logging.getLogger("collect_reviews")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Load .env file (if present) so GITHUB_TOKEN can live in .env
load_dotenv()

GRAPHQL_URL = "https://api.github.com/graphql"
REST_BASE   = "https://api.github.com"

# If remaining REST calls drop below this, sleep until reset.
RATE_LIMIT_BUFFER = int(os.getenv("RATE_LIMIT_BUFFER", "50"))

# Maximum retries for transient errors (5xx / secondary rate-limit).
MAX_RETRIES = 8


# ---------------------------------------------------------------------------
# GitHub Client
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    """Raised when a rate-limit sleep is needed."""


class GitHubClient:
    """Thin wrapper around requests for GitHub REST + GraphQL endpoints.

    Handles:
    - Bearer auth from GITHUB_TOKEN env var
    - Rate-limit tracking (X-RateLimit-* headers)
    - Automatic sleep + retry on 403 / 429 / 5xx
    - Per-request inter-call delay to avoid secondary rate limits
    """

    def __init__(self, token: str, inter_request_delay: float = 0.5) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self.delay = inter_request_delay
        self._last_call_ts: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Enforce minimum inter-request gap."""
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_call_ts = time.monotonic()

    def _check_rate_limit_headers(self, response: requests.Response) -> None:
        """Sleep if remaining quota is critically low."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_at   = response.headers.get("X-RateLimit-Reset")
        if remaining is None:
            return
        remaining = int(remaining)
        if remaining < RATE_LIMIT_BUFFER and reset_at:
            reset_ts  = int(reset_at)
            now_ts    = int(time.time())
            sleep_sec = max(0, reset_ts - now_ts) + 5  # +5 s buffer
            log.warning(
                "Rate-limit quota low (remaining=%d). Sleeping %ds until reset.",
                remaining, sleep_sec,
            )
            time.sleep(sleep_sec)

    def _handle_retry(self, response: requests.Response, attempt: int) -> None:
        """Compute sleep time for retryable errors and sleep."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            sleep_sec = int(retry_after) + 1
        else:
            sleep_sec = min(10 * (2 ** attempt), 300)  # exponential, cap 5 min

        log.warning(
            "HTTP %d on attempt %d/%d. Sleeping %ds before retry.",
            response.status_code, attempt + 1, MAX_RETRIES, sleep_sec,
        )
        time.sleep(sleep_sec)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def rest_get(self, path: str, params: dict | None = None) -> dict | list:
        """GET {REST_BASE}{path} with retry logic. Returns parsed JSON."""
        url = f"{REST_BASE}{path}"
        for attempt in range(MAX_RETRIES):
            self._throttle()
            resp = self._session.get(url, params=params)
            self._check_rate_limit_headers(resp)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (403, 429):
                if attempt < MAX_RETRIES - 1:
                    self._handle_retry(resp, attempt)
                    continue
                resp.raise_for_status()

            if resp.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    self._handle_retry(resp, attempt)
                    continue
                resp.raise_for_status()

            # 4xx that aren't rate limits — raise immediately
            resp.raise_for_status()

        raise RuntimeError(f"Exhausted retries for GET {url}")  # unreachable

    def graphql_query(self, query: str, variables: dict) -> dict:
        """POST to /graphql with retry. Returns the 'data' sub-dict."""
        payload = {"query": query, "variables": variables}
        for attempt in range(MAX_RETRIES):
            self._throttle()
            resp = self._session.post(GRAPHQL_URL, json=payload)
            self._check_rate_limit_headers(resp)

            if resp.status_code in (403, 429):
                if attempt < MAX_RETRIES - 1:
                    self._handle_retry(resp, attempt)
                    continue
                resp.raise_for_status()

            if resp.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    self._handle_retry(resp, attempt)
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            body = resp.json()

            # GraphQL-level errors
            errors = body.get("errors", [])
            if errors:
                for err in errors:
                    etype = err.get("type", "")
                    if etype == "RATE_LIMITED":
                        if attempt < MAX_RETRIES - 1:
                            log.warning("GraphQL RATE_LIMITED. Sleeping 60s.")
                            time.sleep(60)
                            break
                        raise RateLimitError("GraphQL rate limit exhausted.")
                    if etype == "MAX_NODE_LIMIT_EXCEEDED":
                        raise ValueError(
                            "GraphQL node limit exceeded. Reduce --page-size."
                        )
                else:
                    # Non-rate-limit errors
                    raise RuntimeError(f"GraphQL errors: {errors}")
                continue  # retry after RATE_LIMITED sleep

            return body.get("data", {})

        raise RuntimeError("Exhausted retries for GraphQL query")


# ---------------------------------------------------------------------------
# GraphQL query definition
# ---------------------------------------------------------------------------

# We embed reviews + reviewThreads inside the paginated PR query so that
# reviewer participation can be checked with O(pages) calls instead of
# O(PRs) calls.  Patches are fetched separately via REST only for matched PRs.

_PR_DISCOVERY_QUERY = """
query($owner: String!, $repo: String!, $cursor: String, $pageSize: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequests(
      first: $pageSize
      states: [MERGED]
      after: $cursor
      orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        url
        body
        author { login }
        mergedAt
        reviews(first: 100) {
          nodes {
            author { login }
            state
            body
            submittedAt
          }
        }
        reviewThreads(first: 100) {
          nodes {
            comments(first: 50) {
              nodes {
                databaseId
                author { login }
                path
                line
                diffHunk
                body
                createdAt
                replyTo { databaseId }
              }
            }
          }
        }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# PR Discovery (GraphQL)
# ---------------------------------------------------------------------------

def iter_merged_prs(
    client: GitHubClient,
    owner: str,
    repo: str,
    page_size: int = 50,
) -> Iterator[dict]:
    """Yield raw GraphQL PR node dicts for all merged PRs, newest first."""
    cursor: str | None = None
    page_num = 0

    while True:
        page_num += 1
        log.info("Fetching PR page %d (cursor=%s) …", page_num, cursor or "start")
        data = client.graphql_query(
            _PR_DISCOVERY_QUERY,
            variables={
                "owner": owner,
                "repo": repo,
                "cursor": cursor,
                "pageSize": page_size,
            },
        )

        pr_conn = data["repository"]["pullRequests"]
        nodes   = pr_conn["nodes"]
        page_info = pr_conn["pageInfo"]

        for node in nodes:
            yield node

        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]


# ---------------------------------------------------------------------------
# Reviewer Filter
# ---------------------------------------------------------------------------

def reviewer_participated(pr_node: dict, reviewer: str) -> bool:
    """Return True if the target reviewer left any review or comment on this PR."""
    reviewer_lower = reviewer.lower()

    for review in pr_node.get("reviews", {}).get("nodes", []):
        if (review.get("author") or {}).get("login", "").lower() == reviewer_lower:
            return True

    for thread in pr_node.get("reviewThreads", {}).get("nodes", []):
        for comment in thread.get("comments", {}).get("nodes", []):
            if (comment.get("author") or {}).get("login", "").lower() == reviewer_lower:
                return True

    return False


# ---------------------------------------------------------------------------
# Data Extraction
# ---------------------------------------------------------------------------

def extract_review_comments(pr_node: dict, reviewer: str) -> list[dict]:
    """Extract inline review comments from this PR node, filtered to reviewer."""
    reviewer_lower = reviewer.lower()
    comments: list[dict] = []

    for thread in pr_node.get("reviewThreads", {}).get("nodes", []):
        for c in thread.get("comments", {}).get("nodes", []):
            if (c.get("author") or {}).get("login", "").lower() != reviewer_lower:
                continue
            reply_to = c.get("replyTo") or {}
            comments.append({
                "comment_id":     c.get("databaseId"),
                "path":           c.get("path"),
                "line":           c.get("line"),
                "diff_hunk":      c.get("diffHunk", ""),
                "body":           c.get("body", ""),
                "created_at":     c.get("createdAt", ""),
                "in_reply_to_id": reply_to.get("databaseId"),
            })
    return comments


def extract_pr_level_reviews(pr_node: dict, reviewer: str) -> list[dict]:
    """Extract PR-level review summaries filtered to reviewer."""
    reviewer_lower = reviewer.lower()
    reviews: list[dict] = []

    for r in pr_node.get("reviews", {}).get("nodes", []):
        if (r.get("author") or {}).get("login", "").lower() != reviewer_lower:
            continue
        # Skip bare COMMENTED reviews with no body (captured as inline comments).
        state = r.get("state", "")
        body  = (r.get("body") or "").strip()
        if not body and state == "COMMENTED":
            continue
        reviews.append({
            "state":        state,
            "body":         body,
            "submitted_at": r.get("submittedAt", ""),
        })
    return reviews


# ---------------------------------------------------------------------------
# Patch Fetcher (REST)
# ---------------------------------------------------------------------------

def fetch_files(
    client: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
) -> list[dict]:
    """Fetch changed files + patches for a single PR via REST (paginated)."""
    files: list[dict] = []
    page = 1
    while True:
        path = f"/repos/{owner}/{repo}/pulls/{pr_number}/files"
        data = client.rest_get(path, params={"per_page": 100, "page": page})
        if not data:
            break
        for f in data:
            files.append({
                "filename":  f.get("filename", ""),
                "status":    f.get("status", ""),
                "patch":     f.get("patch", ""),   # absent for binary / very large files
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
            })
        if len(data) < 100:
            break
        page += 1
    return files


# ---------------------------------------------------------------------------
# Output Writer — JSON Lines, append-mode, resume-safe
# ---------------------------------------------------------------------------

class JsonlWriter:
    """Append-mode JSON Lines writer with resumability support.

    On construction, reads existing file to populate `seen_prs` so that
    subsequent writes skip already-collected PR numbers.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.seen_prs: set[int] = set()
        self._fh = None
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        log.info("Scanning existing output file for already-collected PRs …")
        count = 0
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
                        count += 1
                except json.JSONDecodeError:
                    pass
        log.info(
            "Found %d already-collected PR(s) — will skip them on this run.",
            count,
        )

    def open(self) -> None:
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        if self._fh is None:
            raise RuntimeError("Writer not opened. Call .open() first.")
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()  # flush after every record so partial runs are safe

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Assemble full PR record
# ---------------------------------------------------------------------------

def build_pr_record(
    pr_node: dict,
    reviewer: str,
    client: GitHubClient,
    owner: str,
    repo: str,
) -> dict:
    """Combine GraphQL data + REST patch data into the output schema."""
    return {
        "pr_number":        pr_node["number"],
        "pr_title":         pr_node.get("title", ""),
        "pr_url":           pr_node.get("url", ""),
        "pr_description":   pr_node.get("body") or "",
        "author":           (pr_node.get("author") or {}).get("login", ""),
        "merged_at":        pr_node.get("mergedAt", ""),
        "files_changed":    fetch_files(client, owner, repo, pr_node["number"]),
        "review_comments":  extract_review_comments(pr_node, reviewer),
        "pr_level_reviews": extract_pr_level_reviews(pr_node, reviewer),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine GitHub PR review history for a specific reviewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo", required=True,
        help='Repository in "owner/name" format, e.g. pandas-dev/pandas.',
    )
    parser.add_argument(
        "--reviewer", required=True,
        help="GitHub username of the target reviewer.",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output JSON Lines file path. "
            "Defaults to {reviewer}_{repo}_reviews.jsonl"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print up to --max-prs matched PRs to stdout; do not write a file.",
    )
    parser.add_argument(
        "--max-prs", type=int, default=None,
        help="Stop after collecting this many matched PRs.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Minimum seconds between API calls (secondary rate-limit guard).",
    )
    parser.add_argument(
        "--page-size", type=int, default=50,
        help=(
            "PRs per GraphQL page (1-100). "
            "Reduce if MAX_NODE_LIMIT_EXCEEDED errors appear."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.getLogger().setLevel(args.log_level)

    # -- Token ----------------------------------------------------------
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        log.error(
            "GITHUB_TOKEN is not set.\n"
            "  Option 1: Paste your token in the .env file next to this script.\n"
            "  Option 2: export GITHUB_TOKEN=ghp_... (or $env:GITHUB_TOKEN in PowerShell)\n"
            "  Create a PAT at: https://github.com/settings/tokens"
        )
        sys.exit(1)

    # -- Parse repo -----------------------------------------------------
    parts = args.repo.strip("/").split("/")
    if len(parts) != 2:
        log.error("--repo must be in 'owner/name' format. Got: %s", args.repo)
        sys.exit(1)
    owner, repo_name = parts

    reviewer = args.reviewer.strip()

    # -- Output file ----------------------------------------------------
    if args.output:
        output_path = Path(args.output)
    else:
        slug = repo_name.replace("/", "_")
        output_path = Path(f"{reviewer}_{slug}_reviews.jsonl")

    # -- Client ---------------------------------------------------------
    client = GitHubClient(token=token, inter_request_delay=args.delay)

    # -- Writer (loads seen PRs for resumability) -----------------------
    writer: JsonlWriter | None = None
    if not args.dry_run:
        writer = JsonlWriter(output_path)
        writer.open()
        skipped_already = len(writer.seen_prs)
    else:
        skipped_already = 0

    # -- Stats ----------------------------------------------------------
    start_time     = time.monotonic()
    prs_scanned    = 0
    prs_skipped    = 0
    prs_matched    = 0
    total_comments = 0
    total_reviews  = 0

    log.info(
        "Starting collection — repo=%s/%s  reviewer=%s  dry_run=%s  max_prs=%s",
        owner, repo_name, reviewer, args.dry_run, args.max_prs,
    )

    try:
        for pr_node in iter_merged_prs(
            client, owner, repo_name, page_size=args.page_size
        ):
            pr_num = pr_node["number"]
            prs_scanned += 1

            # -- Resumability: skip already-written PRs -----------------
            if writer and pr_num in writer.seen_prs:
                log.debug("PR #%d already collected — skipping.", pr_num)
                prs_skipped += 1
                continue

            # -- Filter: did the reviewer participate? ------------------
            if not reviewer_participated(pr_node, reviewer):
                continue

            prs_matched += 1
            log.info(
                "PR #%d matched reviewer '%s' — fetching patches … "
                "(%d matched so far)",
                pr_num, reviewer, prs_matched,
            )

            record = build_pr_record(pr_node, reviewer, client, owner, repo_name)

            n_comments = len(record["review_comments"])
            n_reviews  = len(record["pr_level_reviews"])
            total_comments += n_comments
            total_reviews  += n_reviews

            log.debug(
                "  PR #%d: %d inline comments, %d review summaries, "
                "%d files changed.",
                pr_num, n_comments, n_reviews, len(record["files_changed"]),
            )

            if args.dry_run:
                print(json.dumps(record, ensure_ascii=False, indent=2))
            else:
                writer.write(record)

            # -- Cap at --max-prs ---------------------------------------
            if args.max_prs and prs_matched >= args.max_prs:
                log.info("Reached --max-prs=%d limit. Stopping.", args.max_prs)
                break

    except KeyboardInterrupt:
        log.warning("Interrupted by user (Ctrl-C). Progress saved to %s", output_path)
    finally:
        if writer:
            writer.close()

    # -- Summary --------------------------------------------------------
    elapsed   = time.monotonic() - start_time
    h, rem    = divmod(int(elapsed), 3600)
    m, s      = divmod(rem, 60)
    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"

    summary = [
        "",
        "=" * 52,
        "  Run Complete",
        "=" * 52,
        f"  PRs scanned:            {prs_scanned:>8,}",
        f"  PRs skipped (resume):   {prs_skipped:>8,}",
        f"  PRs matched reviewer:   {prs_matched:>8,}",
        f"  Total inline comments:  {total_comments:>8,}",
        f"  Total review summaries: {total_reviews:>8,}",
        f"  Runtime:                {elapsed_str:>8}",
    ]
    if not args.dry_run:
        summary.append(f"  Output:                 {output_path}")
    summary.append("=" * 52)
    for line in summary:
        log.info(line)


if __name__ == "__main__":
    main()
