#!/usr/bin/env python3
"""
find_reviewer.py — Identify the best candidate "senior reviewer" in a GitHub repo.

Scans the most recent merged PRs, tallies per-user review statistics, and ranks
candidates by a combined score weighting comment volume, PR coverage, and
average comment depth.

Usage:
    python find_reviewer.py \\
        --repo pandas-dev/pandas \\
        [--lookback 500] [--top 10] [--output reviewer_rankings.csv]

Requires:
    GITHUB_TOKEN environment variable or .env file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
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
log = logging.getLogger("find_reviewer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

load_dotenv()

REST_BASE = "https://api.github.com"
RATE_LIMIT_BUFFER = 50
MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# GitHub REST Client (reusable, rate-limit aware)
# ---------------------------------------------------------------------------

class GitHubClient:
    """Minimal REST client with rate-limit backoff and retry."""

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
            "HTTP %d — attempt %d/%d, sleeping %ds.",
            resp.status_code, attempt + 1, MAX_RETRIES, sleep_sec,
        )
        time.sleep(sleep_sec)

    def get(self, path: str, params: dict | None = None) -> list | dict:
        url = f"{REST_BASE}{path}"
        for attempt in range(MAX_RETRIES):
            self._throttle()
            resp = self._session.get(url, params=params)
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
        """Fetch all pages of a list endpoint, up to max_items."""
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


# ---------------------------------------------------------------------------
# SQLite Cache — stores raw API responses keyed by PR number
# ---------------------------------------------------------------------------

class PRCache:
    """Disk-backed cache so re-runs with different scoring don't re-hit the API."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pr_data (
                pr_number  INTEGER PRIMARY KEY,
                repo       TEXT NOT NULL,
                pr_json    TEXT NOT NULL,
                reviews    TEXT NOT NULL,
                comments   TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def has(self, repo: str, pr_number: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM pr_data WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchone()
        return row is not None

    def get(self, repo: str, pr_number: int) -> tuple[dict, list, list] | None:
        row = self._conn.execute(
            "SELECT pr_json, reviews, comments FROM pr_data "
            "WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), json.loads(row[1]), json.loads(row[2])

    def put(
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

    def cached_count(self, repo: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM pr_data WHERE repo = ?", (repo,),
        ).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Per-user statistics
# ---------------------------------------------------------------------------

@dataclass
class ReviewerStats:
    username: str
    review_comment_count: int = 0         # inline diff comments
    formal_review_count: int = 0          # APPROVED / CHANGES_REQUESTED / COMMENTED
    prs_participated: set = field(default_factory=set)
    total_comment_chars: int = 0          # for computing average length
    first_seen: str = ""
    last_seen: str = ""

    @property
    def pr_count(self) -> int:
        return len(self.prs_participated)

    @property
    def avg_comment_length(self) -> float:
        total = self.review_comment_count + self.formal_review_count
        return self.total_comment_chars / total if total > 0 else 0.0

    def update_date(self, date_str: str) -> None:
        if not date_str:
            return
        if not self.first_seen or date_str < self.first_seen:
            self.first_seen = date_str
        if not self.last_seen or date_str > self.last_seen:
            self.last_seen = date_str


# ---------------------------------------------------------------------------
# Bot detection
# ---------------------------------------------------------------------------

def is_bot(user: dict | None) -> bool:
    """Return True if the user object looks like a bot account."""
    if user is None:
        return True
    if user.get("type", "").lower() == "bot":
        return True
    login = user.get("login", "").lower()
    if "bot" in login or login.endswith("[bot]"):
        return True
    return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(stats: ReviewerStats, max_comments: int, max_prs: int) -> float:
    """
    Combined score:
      40% — normalized comment volume   (review_comment_count / max)
      30% — normalized PR coverage       (pr_count / max)
      30% — normalized avg comment length (capped at 500 chars → 1.0)
    """
    vol = stats.review_comment_count / max_comments if max_comments else 0
    cov = stats.pr_count / max_prs if max_prs else 0
    depth = min(stats.avg_comment_length / 500.0, 1.0)
    return 0.40 * vol + 0.30 * cov + 0.30 * depth


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def fetch_pr_data(
    client: GitHubClient,
    cache: PRCache,
    owner: str,
    repo: str,
    lookback: int,
) -> list[tuple[dict, list, list]]:
    """Fetch the last `lookback` merged PRs, using cache where possible."""
    repo_full = f"{owner}/{repo}"

    # Step 1: List merged PRs (sorted by updated, newest first)
    log.info("Listing last %d merged PRs for %s …", lookback, repo_full)
    pr_list = client.get_paginated(
        f"/repos/{owner}/{repo}/pulls",
        per_page=100,
        max_items=lookback,
    )
    # The endpoint returns closed PRs when state=closed, but we passed no
    # state filter — default is "open".  We need state=closed + merged only.
    # Let's re-fetch correctly.
    pr_list = []
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

    log.info("Found %d merged PRs to process.", len(pr_list))

    results: list[tuple[dict, list, list]] = []
    cache_hits = 0

    for i, pr in enumerate(pr_list, 1):
        pr_num = pr["number"]

        cached = cache.get(repo_full, pr_num)
        if cached is not None:
            results.append(cached)
            cache_hits += 1
            continue

        log.info(
            "[%d/%d] PR #%d — fetching reviews + comments …",
            i, len(pr_list), pr_num,
        )

        # Formal reviews
        reviews = client.get_paginated(
            f"/repos/{owner}/{repo}/pulls/{pr_num}/reviews",
            per_page=100,
        )

        # Inline review comments
        comments = client.get_paginated(
            f"/repos/{owner}/{repo}/pulls/{pr_num}/comments",
            per_page=100,
        )

        cache.put(repo_full, pr_num, pr, reviews, comments)
        results.append((pr, reviews, comments))

    log.info(
        "Done fetching. %d from cache, %d from API.",
        cache_hits, len(pr_list) - cache_hits,
    )
    return results


def tally_stats(
    pr_data: list[tuple[dict, list, list]],
) -> dict[str, ReviewerStats]:
    """Aggregate per-user review statistics, excluding bots and self-reviews."""
    stats: dict[str, ReviewerStats] = {}

    for pr, reviews, comments in pr_data:
        pr_num = pr["number"]
        pr_author = (pr.get("user") or {}).get("login", "").lower()

        # -- Formal reviews --
        for rev in reviews:
            user = rev.get("user")
            if is_bot(user):
                continue
            login = user["login"]
            if login.lower() == pr_author:
                continue  # exclude self-reviews

            if login not in stats:
                stats[login] = ReviewerStats(username=login)
            s = stats[login]
            s.formal_review_count += 1
            s.prs_participated.add(pr_num)
            body = (rev.get("body") or "").strip()
            s.total_comment_chars += len(body)
            s.update_date(rev.get("submitted_at", ""))

        # -- Inline review comments --
        for c in comments:
            user = c.get("user")
            if is_bot(user):
                continue
            login = user["login"]
            if login.lower() == pr_author:
                continue

            if login not in stats:
                stats[login] = ReviewerStats(username=login)
            s = stats[login]
            s.review_comment_count += 1
            s.prs_participated.add(pr_num)
            body = (c.get("body") or "").strip()
            s.total_comment_chars += len(body)
            s.update_date(c.get("created_at", ""))

    return stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(ranked: list[tuple[ReviewerStats, float]], top_n: int) -> None:
    """Print a formatted table of the top N reviewers."""
    header = (
        f"{'Rank':>4}  {'Username':<24} {'PRs':>5} {'Comments':>9} "
        f"{'Reviews':>8} {'Avg Len':>8} {'Last Active':<12} {'Score':>6}"
    )
    sep = "-" * len(header)

    print()
    print(sep)
    print("  TOP REVIEWER CANDIDATES")
    print(sep)
    print(header)
    print(sep)

    for i, (s, score) in enumerate(ranked[:top_n], 1):
        last_active = s.last_seen[:10] if s.last_seen else "N/A"
        print(
            f"{i:>4}  {s.username:<24} {s.pr_count:>5} "
            f"{s.review_comment_count:>9} {s.formal_review_count:>8} "
            f"{s.avg_comment_length:>8.0f} {last_active:<12} "
            f"{score:>6.3f}"
        )

    print(sep)
    print()


def write_csv(
    ranked: list[tuple[ReviewerStats, float]],
    output_path: Path,
) -> None:
    """Write the full ranked list to a CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "username", "pr_count", "review_comments",
            "formal_reviews", "avg_comment_length", "first_seen",
            "last_seen", "score",
        ])
        for i, (s, score) in enumerate(ranked, 1):
            writer.writerow([
                i, s.username, s.pr_count, s.review_comment_count,
                s.formal_review_count, f"{s.avg_comment_length:.1f}",
                s.first_seen[:10] if s.first_seen else "",
                s.last_seen[:10] if s.last_seen else "",
                f"{score:.4f}",
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify the best candidate senior reviewer in a GitHub repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo", required=True,
        help='Repository as "owner/name" (e.g. pandas-dev/pandas).',
    )
    parser.add_argument(
        "--lookback", type=int, default=500,
        help="Number of most recent merged PRs to scan.",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="How many top candidates to display.",
    )
    parser.add_argument(
        "--output", default=None,
        help="CSV output file. Defaults to {repo_name}_reviewers.csv.",
    )
    parser.add_argument(
        "--cache-dir", default=".reviewer_cache",
        help="Directory for the SQLite cache DB.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Minimum seconds between API calls.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    # -- Token ---
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        log.error(
            "GITHUB_TOKEN is not set.\n"
            "  Option 1: Paste your token in the .env file.\n"
            "  Option 2: export GITHUB_TOKEN=ghp_... "
            "(or $env:GITHUB_TOKEN in PowerShell)\n"
            "  Create a PAT at: https://github.com/settings/tokens"
        )
        sys.exit(1)

    # -- Repo ---
    parts = args.repo.strip("/").split("/")
    if len(parts) != 2:
        log.error("--repo must be 'owner/name'. Got: %s", args.repo)
        sys.exit(1)
    owner, repo_name = parts

    # -- Output ---
    if args.output:
        csv_path = Path(args.output)
    else:
        csv_path = Path(f"{repo_name}_reviewers.csv")

    # -- Cache ---
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_db = cache_dir / f"{owner}_{repo_name}.sqlite"
    cache = PRCache(cache_db)
    cached_before = cache.cached_count(f"{owner}/{repo_name}")
    if cached_before:
        log.info("Cache has %d PRs already — will skip API calls for those.", cached_before)

    # -- Client ---
    client = GitHubClient(token=token, delay=args.delay)

    # -- Fetch ---
    start = time.monotonic()
    try:
        pr_data = fetch_pr_data(client, cache, owner, repo_name, args.lookback)
    except KeyboardInterrupt:
        log.warning("Interrupted. Partial results cached to %s", cache_db)
        cache.close()
        sys.exit(130)

    # -- Tally ---
    stats = tally_stats(pr_data)

    if not stats:
        log.warning("No reviewer activity found in %d PRs.", len(pr_data))
        cache.close()
        return

    # -- Score & rank ---
    max_comments = max(s.review_comment_count for s in stats.values())
    max_prs = max(s.pr_count for s in stats.values())

    ranked: list[tuple[ReviewerStats, float]] = []
    for s in stats.values():
        score = compute_score(s, max_comments, max_prs)
        ranked.append((s, score))
    ranked.sort(key=lambda x: x[1], reverse=True)

    # -- Output ---
    print_table(ranked, args.top)
    write_csv(ranked, csv_path)
    log.info("Full rankings written to %s", csv_path)

    elapsed = time.monotonic() - start
    m, s_rem = divmod(int(elapsed), 60)
    log.info(
        "Done. Scanned %d PRs, found %d reviewers, runtime %d:%02d.",
        len(pr_data), len(stats), m, s_rem,
    )

    cache.close()


if __name__ == "__main__":
    main()
