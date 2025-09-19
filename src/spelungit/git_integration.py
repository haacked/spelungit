"""
Git repository integration for extracting commit data.
Provides methods to get commit information, diffs, and history.
"""

import asyncio
import logging
import re
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GitRepository:
    """Git repository wrapper for extracting commit data."""

    def __init__(self, repository_path: str):
        self.repository_path = repository_path

    async def _run_git_command(self, args: List[str], timeout: int = 30) -> str:
        """Run a git command and return output."""
        cmd = ["git", "-C", self.repository_path] + args

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode or -1, cmd, output=stdout, stderr=stderr
                )

            return stdout.decode("utf-8", errors="ignore")

        except asyncio.TimeoutError:
            logger.error(f"Git command timed out: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Git command failed: {' '.join(cmd)} - {e}")
            raise

    async def get_commit_count(self) -> int:
        """Get the total number of commits in the repository."""
        try:
            output = await self._run_git_command(["rev-list", "--count", "HEAD"])
            return int(output.strip())
        except Exception as e:
            logger.warning(f"Could not get commit count: {e}")
            return 0

    async def get_commits_since(
        self, since_date: Optional[datetime] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get commits since a specific date."""
        args = ["log", "--format=%H|%an|%ae|%at|%s", "--no-merges"]

        if since_date:
            args.append(f"--since={since_date.isoformat()}")

        if limit:
            args.append(f"-{limit}")

        try:
            output = await self._run_git_command(args)
            commits = []

            for line in output.strip().split("\n"):
                if not line:
                    continue

                parts = line.split("|", 4)
                if len(parts) >= 5:
                    sha, author_name, author_email, timestamp, message = parts

                    # Parse timestamp
                    try:
                        commit_date = datetime.fromtimestamp(int(timestamp))
                    except (ValueError, OSError):
                        commit_date = datetime.now()

                    # Get additional commit info
                    commit_info = await self.get_commit_info(sha)

                    commits.append(
                        {
                            "sha": sha,
                            "author": author_name,
                            "author_email": author_email,
                            "date": commit_date,
                            "message": message,
                            "authors": commit_info.get("authors", [author_name]),
                            "files_changed": commit_info.get("files_changed", []),
                            "diff": commit_info.get("diff", ""),
                        }
                    )

            return commits

        except Exception as e:
            logger.error(f"Failed to get commits since {since_date}: {e}")
            return []

    async def get_commit_info(self, sha: str) -> Dict[str, Any]:
        """Get detailed information about a specific commit."""
        try:
            # Get commit metadata
            format_str = "%H|%an|%ae|%at|%s|%b"
            metadata_output = await self._run_git_command(
                ["show", "--format=" + format_str, "--no-patch", sha]
            )

            # Parse metadata
            lines = metadata_output.strip().split("\n")
            if not lines[0]:
                raise ValueError(f"No commit found with SHA: {sha}")

            parts = lines[0].split("|", 5)
            if len(parts) < 6:
                raise ValueError(f"Invalid commit format for SHA: {sha}")

            commit_sha, author_name, author_email, timestamp, subject, body = parts

            # Parse timestamp
            try:
                commit_date = datetime.fromtimestamp(int(timestamp))
            except (ValueError, OSError):
                commit_date = datetime.now()

            # Extract co-authors from commit message
            full_message = subject + ("\n" + body if body.strip() else "")
            authors = self._extract_authors(full_message, author_name)

            # Get files changed
            try:
                files_output = await self._run_git_command(
                    ["show", "--name-only", "--format=", sha]
                )
                files_changed = [f.strip() for f in files_output.strip().split("\n") if f.strip()]
            except Exception:
                files_changed = []

            # Get diff (limited to avoid huge outputs)
            try:
                diff_output = await self._run_git_command(["show", "--format=", "--unified=3", sha])
                # Limit diff size to prevent memory issues
                if len(diff_output) > 50000:  # ~50KB limit
                    diff_output = diff_output[:50000] + "\n... (diff truncated)"
            except Exception:
                diff_output = ""

            return {
                "sha": commit_sha,
                "author": author_name,
                "author_email": author_email,
                "date": commit_date.isoformat(),
                "message": subject,
                "full_message": full_message,
                "authors": authors,
                "files_changed": files_changed,
                "diff": diff_output,
            }

        except Exception as e:
            logger.error(f"Failed to get commit info for {sha}: {e}")
            return {
                "sha": sha,
                "author": "Unknown",
                "author_email": "",
                "date": datetime.now().isoformat(),
                "message": "Could not retrieve commit information",
                "full_message": "",
                "authors": [],
                "files_changed": [],
                "diff": "",
            }

    def _extract_authors(self, commit_message: str, primary_author: str) -> List[str]:
        """Extract all authors from commit message, including Co-authored-by trailers."""
        authors = [primary_author]

        # Look for Co-authored-by trailers
        co_author_pattern = r"Co-authored-by:\s*([^<\n]+)(?:\s*<[^>]*>)?"

        for match in re.finditer(co_author_pattern, commit_message, re.IGNORECASE):
            co_author = match.group(1).strip()
            if co_author and co_author not in authors:
                authors.append(co_author)

        return authors

    async def get_recent_commits(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent commits."""
        return await self.get_commits_since(limit=limit)

    async def is_valid_repository(self) -> bool:
        """Check if the path is a valid Git repository."""
        try:
            await self._run_git_command(["rev-parse", "--git-dir"])
            return True
        except Exception:
            return False

    async def get_repository_stats(self) -> Dict[str, Any]:
        """Get basic repository statistics."""
        try:
            # Get commit count
            commit_count = await self.get_commit_count()

            # Get branch info
            try:
                current_branch = await self._run_git_command(["branch", "--show-current"])
                current_branch = current_branch.strip()
            except Exception:
                current_branch = "unknown"

            # Get latest commit
            try:
                latest_commit = await self._run_git_command(["log", "-1", "--format=%H|%an|%at|%s"])
                if latest_commit.strip():
                    parts = latest_commit.strip().split("|", 3)
                    if len(parts) >= 4:
                        latest_sha, latest_author, latest_timestamp, latest_message = parts
                        try:
                            latest_date = datetime.fromtimestamp(int(latest_timestamp))
                        except (ValueError, OSError):
                            latest_date = datetime.now()
                    else:
                        latest_sha, latest_author, latest_date, latest_message = (
                            "unknown",
                            "unknown",
                            datetime.now(),
                            "unknown",
                        )
                else:
                    latest_sha, latest_author, latest_date, latest_message = (
                        "unknown",
                        "unknown",
                        datetime.now(),
                        "unknown",
                    )
            except Exception:
                latest_sha, latest_author, latest_date, latest_message = (
                    "unknown",
                    "unknown",
                    datetime.now(),
                    "unknown",
                )

            return {
                "commit_count": commit_count,
                "current_branch": current_branch,
                "latest_commit": {
                    "sha": latest_sha,
                    "author": latest_author,
                    "date": latest_date.isoformat()
                    if isinstance(latest_date, datetime)
                    else str(latest_date),
                    "message": latest_message,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get repository stats: {e}")
            return {
                "commit_count": 0,
                "current_branch": "unknown",
                "latest_commit": {
                    "sha": "unknown",
                    "author": "unknown",
                    "date": datetime.now().isoformat(),
                    "message": "unknown",
                },
            }

    async def get_blame_info(
        self,
        file_path: str,
        commit_sha: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get blame information for a file, showing who wrote which lines.

        Args:
            file_path: Path to the file relative to repository root
            commit_sha: Optional commit SHA to blame at specific point in history
            start_line: Optional starting line number (1-indexed)
            end_line: Optional ending line number (1-indexed)

        Returns:
            List of blame entries with line info, author, commit, and content
        """
        try:
            # Build git blame command
            blame_args = ["blame", "--porcelain"]

            if start_line and end_line:
                blame_args.extend(["-L", f"{start_line},{end_line}"])
            elif start_line:
                blame_args.extend(["-L", f"{start_line},+10"])  # Show 10 lines from start

            if commit_sha:
                blame_args.append(commit_sha)

            blame_args.append(file_path)

            output = await self._run_git_command(blame_args)

            # Parse porcelain blame output
            blame_entries = []
            lines = output.split("\n")
            i = 0

            while i < len(lines):
                if not lines[i].strip():
                    i += 1
                    continue

                # First line: commit SHA, original line, final line, group size
                header_parts = lines[i].split()
                if len(header_parts) < 3:
                    i += 1
                    continue

                commit_sha = header_parts[0]
                int(header_parts[1])
                final_line = int(header_parts[2])

                # Parse metadata lines until we hit the content line (starts with tab)
                i += 1
                author = "Unknown"
                author_mail = ""
                author_time = ""
                summary = ""

                while i < len(lines) and not lines[i].startswith("\t"):
                    line = lines[i].strip()
                    if line.startswith("author "):
                        author = line[7:]  # Remove 'author ' prefix
                    elif line.startswith("author-mail "):
                        author_mail = line[12:].strip("<>")
                    elif line.startswith("author-time "):
                        author_time = line[12:]
                    elif line.startswith("committer "):
                        line[10:]
                    elif line.startswith("committer-time "):
                        line[15:]
                    elif line.startswith("summary "):
                        summary = line[8:]
                    i += 1

                # Get the actual line content (starts with tab)
                content = ""
                if i < len(lines) and lines[i].startswith("\t"):
                    content = lines[i][1:]  # Remove leading tab
                    i += 1

                # Convert timestamp to readable format
                try:
                    if author_time:
                        dt = datetime.fromtimestamp(int(author_time))
                        formatted_time = dt.isoformat()
                    else:
                        formatted_time = "unknown"
                except (ValueError, OSError):
                    formatted_time = "unknown"

                blame_entries.append(
                    {
                        "line_number": final_line,
                        "commit_sha": commit_sha,
                        "author": author,
                        "author_email": author_mail,
                        "date": formatted_time,
                        "summary": summary,
                        "content": content,
                        "file_path": file_path,
                    }
                )

            return blame_entries

        except subprocess.CalledProcessError as e:
            if "no such path" in e.stderr.decode().lower():
                logger.warning(f"File not found for blame: {file_path}")
                return []
            elif "binary file" in e.stderr.decode().lower():
                logger.warning(f"Cannot blame binary file: {file_path}")
                return []
            else:
                logger.error(f"Git blame failed for {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error getting blame info for {file_path}: {e}")
            raise

    async def get_file_content_at_commit(self, commit_sha: str, file_path: str) -> Optional[str]:
        """Get file content at a specific commit.

        Args:
            commit_sha: Commit SHA to get file content from
            file_path: Path to file relative to repository root

        Returns:
            File content as string, or None if file doesn't exist
        """
        try:
            output = await self._run_git_command(["show", f"{commit_sha}:{file_path}"])
            return output
        except subprocess.CalledProcessError as e:
            if (
                "does not exist" in e.stderr.decode().lower()
                or "path not in" in e.stderr.decode().lower()
            ):
                return None
            else:
                logger.error(f"Error getting file content for {file_path} at {commit_sha}: {e}")
                raise
