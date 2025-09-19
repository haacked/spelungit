"""
Git repository scanner for efficient commit discovery and processing.
Provides utilities for incremental indexing and batch processing.
"""

import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class GitScanner:
    """Efficient Git repository scanner for commit discovery."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        if not self._validate_repo():
            raise ValueError(f"Invalid Git repository: {repo_path}")

    def _validate_repo(self) -> bool:
        """Validate that the path is a Git repository."""
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_repository_stats(self) -> Dict:
        """Get basic repository statistics quickly."""
        try:
            # Get total commit count
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            total_commits = int(result.stdout.strip())

            # Get date range
            result = subprocess.run(
                ["git", "log", "--pretty=format:%ct", "--max-count=1"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            latest_timestamp = int(result.stdout.strip())
            latest_date = datetime.fromtimestamp(latest_timestamp, tz=timezone.utc)

            # Get earliest commit
            result = subprocess.run(
                ["git", "log", "--reverse", "--pretty=format:%ct", "--max-count=1"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            earliest_timestamp = int(result.stdout.strip())
            earliest_date = datetime.fromtimestamp(earliest_timestamp, tz=timezone.utc)

            return {
                "repository_path": str(self.repo_path),
                "total_commits": total_commits,
                "latest_commit_date": latest_date,
                "earliest_commit_date": earliest_date,
            }

        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Error getting repository stats: {e}")
            raise

    def get_all_commit_shas(self) -> List[str]:
        """Get all commit SHAs in the repository, newest first."""
        try:
            result = subprocess.run(
                ["git", "rev-list", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().split("\n")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting commit SHAs: {e}")
            raise

    def get_commits_since_date(self, since_date: datetime) -> List[str]:
        """Get commit SHAs since a specific date."""
        try:
            timestamp = int(since_date.timestamp())
            result = subprocess.run(
                ["git", "rev-list", f"--since={timestamp}", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            if not result.stdout.strip():
                return []

            return result.stdout.strip().split("\n")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting commits since {since_date}: {e}")
            raise

    def get_commit_details(self, sha: str) -> Dict:
        """Get detailed information about a specific commit."""
        try:
            # Get commit metadata
            result = subprocess.run(
                ["git", "show", sha, "--pretty=format:%H|%s|%an|%ae|%ct|%B", "-s"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            lines = result.stdout.strip().split("\n")
            first_line = lines[0]
            parts = first_line.split("|", 5)

            if len(parts) < 6:
                raise ValueError(f"Unexpected git output format for {sha}")

            full_sha, subject, author_name, author_email, timestamp_str, full_message = parts
            timestamp = int(timestamp_str)
            commit_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)

            # Extract co-authors from full message
            co_authors = self._extract_co_authors(full_message)

            # Get files changed
            result = subprocess.run(
                ["git", "show", sha, "--pretty=format:", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            files_changed = [line for line in result.stdout.strip().split("\n") if line.strip()]

            # Get diff with context
            result = subprocess.run(
                ["git", "show", sha, "--pretty=format:", "--unified=3"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            diff_content = result.stdout.strip()

            return {
                "sha": full_sha,
                "message": full_message.strip(),
                "subject": subject,
                "author_name": author_name,
                "author_email": author_email,
                "commit_date": commit_date,
                "co_authors": co_authors,
                "files_changed": files_changed,
                "diff": diff_content,
            }

        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Error getting commit details for {sha}: {e}")
            raise

    def get_commits_batch_details(self, shas: List[str], batch_size: int = 50) -> List[Dict]:
        """Get details for multiple commits efficiently."""
        results = []

        for i in range(0, len(shas), batch_size):
            batch = shas[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch)} commits")

            for sha in batch:
                try:
                    commit_details = self.get_commit_details(sha)
                    results.append(commit_details)
                except Exception as e:
                    logger.error(f"Failed to get details for commit {sha}: {e}")
                    continue

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

    def find_commits_with_coauthors(self, limit: int = 100) -> List[str]:
        """Find commits that have co-authors."""
        try:
            result = subprocess.run(
                [
                    "git",
                    "log",
                    "--grep=Co-authored-by",
                    f"--max-count={limit}",
                    "--pretty=format:%H",
                ],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            if not result.stdout.strip():
                return []

            return result.stdout.strip().split("\n")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error finding co-authored commits: {e}")
            return []

    def get_sample_commits(self, count: int = 100) -> List[str]:
        """Get a sample of recent commits for testing."""
        try:
            result = subprocess.run(
                ["git", "rev-list", "HEAD", f"--max-count={count}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            if not result.stdout.strip():
                return []

            return result.stdout.strip().split("\n")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting sample commits: {e}")
            return []
