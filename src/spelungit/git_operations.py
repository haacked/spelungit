"""Git operations for fetching commit data."""
# type: ignore  # Legacy file - ignore mypy errors

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional

import git
from git.objects.commit import Commit

from .models import CommitInfo

logger = logging.getLogger(__name__)


class GitManager:
    """Manages Git operations for fetching commit information."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.repo: Optional[git.Repo] = None

    async def initialize(self) -> None:
        """Initialize Git repository connection."""
        try:
            self.repo = git.Repo(self.repo_path)
            logger.info(f"Initialized Git repository at {self.repo_path}")
        except Exception as e:
            raise ValueError(f"Failed to initialize Git repository: {e}")

    async def get_commit_info(self, sha: str) -> Optional[CommitInfo]:
        """Get detailed information about a specific commit."""
        if not self.repo:
            raise RuntimeError("Git repository not initialized")

        try:
            commit: Commit = self.repo.commit(sha)

            # Get commit diff
            if commit.parents:
                # Compare with first parent
                diff = self.repo.git.diff(
                    commit.parents[0].hexsha,
                    commit.hexsha,
                    unified=3,  # 3 lines of context
                )
            else:
                # Initial commit - show full files
                diff = self.repo.git.show(commit.hexsha, format="", unified=3)

            # Parse co-authors from commit message
            co_authors = self._extract_co_authors(commit.message)

            # Get files changed
            if commit.parents:
                files_changed = (
                    list(commit.parents[0].diff(commit).iter_change_type("A"))
                    + list(commit.parents[0].diff(commit).iter_change_type("D"))
                    + list(commit.parents[0].diff(commit).iter_change_type("M"))
                )
                files_changed = [item.a_path or item.b_path for item in files_changed]
            else:
                # Initial commit
                files_changed = list(commit.stats.files.keys())

            return CommitInfo(
                sha=commit.hexsha,
                message=commit.message.strip(),
                author_name=commit.author.name,
                author_email=commit.author.email,
                commit_date=datetime.fromtimestamp(commit.committed_date),
                files_changed=files_changed,
                diff=diff,
                co_authors=co_authors,
            )

        except Exception as e:
            logger.error(f"Error fetching commit info for {sha}: {e}")
            return None

    async def get_all_commit_shas(self) -> List[str]:
        """Get all commit SHAs in the repository."""
        if not self.repo:
            raise RuntimeError("Git repository not initialized")

        try:
            # Get all commits in reverse chronological order
            commits = list(self.repo.iter_commits("--all"))
            return [commit.hexsha for commit in commits]

        except Exception as e:
            logger.error(f"Error fetching commit SHAs: {e}")
            raise

    async def get_commits_since(self, since_date: datetime) -> List[str]:
        """Get commit SHAs since a specific date."""
        if not self.repo:
            raise RuntimeError("Git repository not initialized")

        try:
            since_timestamp = int(since_date.timestamp())
            commits = list(self.repo.iter_commits("--all", since=since_timestamp))
            return [commit.hexsha for commit in commits]

        except Exception as e:
            logger.error(f"Error fetching commits since {since_date}: {e}")
            raise

    async def get_commit_batch_info(self, shas: List[str]) -> List[CommitInfo]:
        """Get commit info for multiple SHAs in parallel."""
        if not shas:
            return []

        # Process in smaller batches to avoid overwhelming Git
        batch_size = 10
        results = []

        for i in range(0, len(shas), batch_size):
            batch = shas[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[self.get_commit_info(sha) for sha in batch], return_exceptions=True
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch commit info: {result}")
                elif result is not None:
                    results.append(result)

        return results

    def _extract_co_authors(self, commit_message: str) -> List[str]:
        """Extract co-authors from commit message trailers."""
        co_authors = []

        # Look for Co-authored-by trailers
        co_author_pattern = r"Co-authored-by:\s*(.+?)\s*<(.+?)>"
        matches = re.findall(co_author_pattern, commit_message, re.IGNORECASE | re.MULTILINE)

        for name, email in matches:
            co_authors.append(f"{name.strip()} <{email.strip()}>")

        return co_authors

    async def get_repository_stats(self) -> dict:
        """Get basic statistics about the repository."""
        if not self.repo:
            raise RuntimeError("Git repository not initialized")

        try:
            total_commits = len(list(self.repo.iter_commits("--all")))

            # Get latest commit date
            latest_commit = next(self.repo.iter_commits("--all"))
            latest_date = datetime.fromtimestamp(latest_commit.committed_date)

            # Get earliest commit date
            commits_list = list(self.repo.iter_commits("--all"))
            earliest_commit = commits_list[-1] if commits_list else latest_commit
            earliest_date = datetime.fromtimestamp(earliest_commit.committed_date)

            return {
                "total_commits": total_commits,
                "latest_commit_date": latest_date,
                "earliest_commit_date": earliest_date,
                "repository_path": self.repo_path,
            }

        except Exception as e:
            logger.error(f"Error getting repository stats: {e}")
            raise
