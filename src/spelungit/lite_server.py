#!/usr/bin/env python3
"""
Lite MCP server for Git History search using SQLite + sentence-transformers.
Zero-config deployment without PostgreSQL or OpenAI dependencies.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from pydantic import AnyUrl
except ImportError:
    # Fallback for type annotations
    AnyUrl = str  # type: ignore

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.lowlevel.server import NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import Resource, TextContent, Tool

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

    # Mock classes for development/testing
    class MockServer:
        def __init__(self, name: str):
            self.name = name
            self._resources: Dict[str, Any] = {}
            self._tools: Dict[str, Any] = {}

        def list_resources(self):
            def decorator(func):
                self._resources["list"] = func
                return func

            return decorator

        def read_resource(self):
            def decorator(func):
                self._resources["read"] = func
                return func

            return decorator

        def list_tools(self):
            def decorator(func):
                self._tools["list"] = func
                return func

            return decorator

        def call_tool(self):
            def decorator(func):
                self._tools["call"] = func
                return func

            return decorator

    # Mock MCP types
    class MockResource:
        def __init__(self, uri, name, description, mimeType):
            self.uri = uri
            self.name = name
            self.description = description
            self.mimeType = mimeType

    class MockTool:
        def __init__(self, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

    class MockTextContent:
        def __init__(self, type, text):
            self.type = type
            self.text = text

    # Use mock classes when MCP is not available
    Server = MockServer  # type: ignore
    Resource = MockResource  # type: ignore
    Tool = MockTool  # type: ignore
    TextContent = MockTextContent  # type: ignore


# Add src to path for local development
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir / "src"))

from spelungit.exceptions import (  # noqa: E402
    RepositoryIndexingException,
    RepositoryNotIndexedException,
)
from spelungit.git_integration import GitRepository  # noqa: E402
from spelungit.lite_embeddings import LiteEmbeddingManager  # noqa: E402
from spelungit.models import RepositoryStatus  # noqa: E402
from spelungit.repository_utils import (  # noqa: E402
    generate_repository_id,
    get_canonical_repository_path,
    get_repository_info,
    validate_repository_path,
)
from spelungit.sqlite_database import SQLiteDatabaseManager  # noqa: E402

logger = logging.getLogger(__name__)

# Global instances
db_manager = None
embedding_manager = None


class LiteSearchEngine:
    """Lite search engine using SQLite + sentence-transformers."""

    def __init__(self, db_manager: SQLiteDatabaseManager, embedding_manager: LiteEmbeddingManager):
        self.db = db_manager
        self.embeddings = embedding_manager
        # Cache for staleness checks to avoid repeated git calls
        self._staleness_cache = {}
        # Track active background indexing tasks to prevent race conditions
        self._background_tasks = {}
        # Track progress and metadata for background tasks
        self._background_progress = {}
        # Configuration for auto-updates
        self.background_threshold = 50  # Commits threshold for background vs foreground indexing
        self.staleness_check_cache_minutes = 5  # Cache validity in minutes
        self.enable_auto_update = True  # Global enable/disable flag

    async def _detect_repository_context(self, repository_path: Optional[str] = None) -> tuple:
        """Detect which repository to search based on context."""
        if not repository_path:
            # Use current working directory
            repository_path = os.getcwd()

        if not validate_repository_path(repository_path):
            raise ValueError(f"Path is not a valid Git repository: {repository_path}")

        canonical_path = get_canonical_repository_path(repository_path)
        repository_id = generate_repository_id(canonical_path)

        # Get or create repository record
        repository = await self.db.get_or_create_repository(repository_id, canonical_path)

        return repository_id, repository

    async def _with_git_repo(self, repo_path: str, operation):
        """Execute an operation with a GitRepository instance, ensuring proper resource management."""
        git_repo = None
        try:
            git_repo = GitRepository(repo_path)
            return await operation(git_repo)
        except Exception as e:
            logger.warning(f"Git operation failed for {repo_path}: {e}")
            raise
        finally:
            # GitRepository uses subprocess.communicate() which properly cleans up,
            # but we ensure no lingering references
            git_repo = None

    async def _is_index_stale(self, repository_id: str, repo_path: str) -> Tuple[bool, int]:
        """Check if the index is stale and return commit gap count."""
        cache_key = repository_id
        now = datetime.now()

        # Check cache first
        if cache_key in self._staleness_cache:
            cached_data, cache_time = self._staleness_cache[cache_key]
            cache_age_minutes = (now - cache_time).total_seconds() / 60
            if cache_age_minutes < self.staleness_check_cache_minutes:
                return cached_data

        try:
            # Get latest indexed commit SHA
            latest_indexed_sha = await self.db.get_latest_commit_sha(repository_id)
            if not latest_indexed_sha:
                # No commits indexed yet
                result = (True, -1)  # -1 indicates full index needed
                self._staleness_cache[cache_key] = (result, now)
                return result

            # Count commits between latest indexed SHA and HEAD
            async def count_commits(git_repo):
                try:
                    # Use SHA-based range: count commits from latest_sha..HEAD
                    output = await git_repo._run_git_command(
                        ["rev-list", "--count", f"{latest_indexed_sha}..HEAD"]
                    )
                    return int(output.strip()) if output.strip() else 0
                except Exception as e:
                    logger.warning(f"Failed to count commits using SHA range: {e}")
                    # Fallback: just check if HEAD exists and is different
                    try:
                        head_sha = await git_repo._run_git_command(["rev-parse", "HEAD"])
                        head_sha = head_sha.strip()
                        if head_sha == latest_indexed_sha:
                            return 0
                        else:
                            # We know there's at least one new commit, but can't count exactly
                            return 1
                    except Exception:
                        return 0

            commit_gap = await self._with_git_repo(repo_path, count_commits)

            result = (commit_gap > 0, commit_gap)
            self._staleness_cache[cache_key] = (result, now)
            return result

        except Exception as e:
            logger.warning(f"Error checking index staleness for {repository_id}: {e}")
            # On error, assume not stale to avoid disruption
            result = (False, 0)
            self._staleness_cache[cache_key] = (result, now)
            return result

    async def _start_background_indexing(self, repository_id: str, repository) -> None:
        """Start background indexing for large commit gaps."""
        try:
            logger.info(f"Starting background indexing for {repository_id}")

            # Initialize progress tracking
            self._background_progress[repository_id] = {
                "started_at": datetime.now(),
                "phase": "initializing",
                "commits_processed": 0,
                "total_commits": 0,
                "current_batch": 0,
                "total_batches": 0,
                "last_updated": datetime.now(),
            }

            await self.db.update_repository_status(
                repository_id, RepositoryStatus.INDEXING, progress=0
            )

            async def get_commits(git_repo):
                latest_date = await self.db.get_latest_commit_date(repository_id)
                return await git_repo.get_commits_since(latest_date)

            commits = await self._with_git_repo(repository.canonical_path, get_commits)

            if not commits:
                # Update progress tracking for empty commit set
                self._background_progress[repository_id].update(
                    {"phase": "completed", "last_updated": datetime.now()}
                )
                await self.db.update_repository_status(
                    repository_id,
                    RepositoryStatus.INDEXED,
                    commit_count=await self.db.get_commit_count(repository_id),
                )
                # Clean up progress tracking
                if repository_id in self._background_progress:
                    del self._background_progress[repository_id]
                return

            total_commits = len(commits)
            batch_size = 20  # Smaller batches for background processing
            total_batches = (total_commits + batch_size - 1) // batch_size

            # Update progress tracking with commit details
            self._background_progress[repository_id].update(
                {
                    "phase": "processing",
                    "total_commits": total_commits,
                    "total_batches": total_batches,
                    "last_updated": datetime.now(),
                }
            )

            # Process commits in batches
            for i in range(0, total_commits, batch_size):
                batch = commits[i : i + batch_size]

                for commit_data in batch:
                    if await self.db.commit_exists(repository_id, commit_data["sha"]):
                        continue

                    content = self.embeddings.format_commit_for_embedding(
                        message=commit_data["message"],
                        diff=commit_data.get("diff", ""),
                        files_changed=commit_data.get("files_changed", []),
                    )

                    embedding = await self.embeddings.generate_embedding(
                        content, files_changed=commit_data.get("files_changed", [])
                    )

                    from spelungit.models import StoredCommit

                    stored_commit = StoredCommit(
                        repository_id=repository_id,
                        sha=commit_data["sha"],
                        embedding=embedding,
                        commit_date=commit_data["date"],
                        created_at=datetime.now(),
                    )

                    authors = commit_data.get("authors", [commit_data.get("author", "")])
                    await self.db.store_commit(
                        stored_commit,
                        authors,
                        message=commit_data.get("message", ""),
                        diff_content=commit_data.get("diff", ""),
                    )

                # Update progress
                processed = min(i + batch_size, total_commits)
                progress = int((processed / total_commits) * 100)
                current_batch = (i // batch_size) + 1

                # Update detailed progress tracking
                self._background_progress[repository_id].update(
                    {
                        "commits_processed": processed,
                        "current_batch": current_batch,
                        "progress_percent": progress,
                        "last_updated": datetime.now(),
                    }
                )

                await self.db.update_repository_status(
                    repository_id, RepositoryStatus.INDEXING, progress=progress
                )

                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.1)

            # Mark as completed
            total_count = await self.db.get_commit_count(repository_id)
            await self.db.update_repository_status(
                repository_id, RepositoryStatus.INDEXED, commit_count=total_count
            )

            # Clear cache for this repository
            if repository_id in self._staleness_cache:
                del self._staleness_cache[repository_id]

            # Mark progress as completed and calculate duration
            if repository_id in self._background_progress:
                progress_info = self._background_progress[repository_id]
                duration = datetime.now() - progress_info["started_at"]
                logger.info(
                    f"✅ Background indexing completed for {repository_id}: "
                    f"{len(commits)} commits in {duration.total_seconds():.1f}s"
                )
                # Clean up progress tracking
                del self._background_progress[repository_id]
            else:
                logger.info(
                    f"✅ Background indexing completed for {repository_id}: {len(commits)} commits"
                )

        except Exception as e:
            logger.error(f"Background indexing failed for {repository_id}: {e}")

            # Determine if we should preserve partial progress or revert completely
            commits_processed = 0
            if repository_id in self._background_progress:
                commits_processed = self._background_progress[repository_id].get(
                    "commits_processed", 0
                )

            # Preserve partial progress if some commits were successfully indexed
            if commits_processed > 0:
                logger.info(f"Preserving partial progress: {commits_processed} commits indexed")
                # Update repository status with current count and error message
                total_count = await self.db.get_commit_count(repository_id)
                await self.db.update_repository_status(
                    repository_id,
                    RepositoryStatus.INDEXED,
                    commit_count=total_count,
                    error_message=f"Partial indexing completed: {commits_processed} commits. Error: {str(e)[:200]}",
                )
            else:
                # No progress made, revert to previous state
                await self.db.update_repository_status(
                    repository_id,
                    RepositoryStatus.INDEXED,  # Revert to previous status
                    error_message=f"Background indexing failed: {str(e)[:200]}",
                )

            # Clear staleness cache to force re-evaluation
            if repository_id in self._staleness_cache:
                del self._staleness_cache[repository_id]

            # Clean up progress tracking on error
            if repository_id in self._background_progress:
                del self._background_progress[repository_id]
        finally:
            # Clean up the task reference
            if repository_id in self._background_tasks:
                del self._background_tasks[repository_id]

    def _is_transient_error(self, error: Exception) -> bool:
        """Determine if an error is likely transient and worth retrying."""
        error_str = str(error).lower()
        transient_indicators = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "unavailable",
            "busy",
            "locked",
            "permission denied",
        ]
        return any(indicator in error_str for indicator in transient_indicators)

    async def _start_background_indexing_with_retry(
        self, repository_id: str, repository, max_retries: int = 2, base_delay: float = 2.0
    ) -> None:
        """Start background indexing with retry logic for transient failures."""
        for attempt in range(max_retries + 1):
            try:
                await self._start_background_indexing(repository_id, repository)
                return  # Success, no need to retry
            except Exception as e:
                if attempt == max_retries or not self._is_transient_error(e):
                    # Final attempt or non-transient error, re-raise
                    raise

                # Wait before retrying with exponential backoff
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Background indexing attempt {attempt + 1} failed for {repository_id}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

    def get_background_task_progress(self, repository_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed progress information for a background indexing task."""
        if repository_id not in self._background_progress:
            return None

        progress_info = self._background_progress[repository_id].copy()

        # Calculate elapsed time
        elapsed = datetime.now() - progress_info["started_at"]
        progress_info["elapsed_seconds"] = elapsed.total_seconds()

        # Calculate estimated time remaining (if we have processed some commits)
        if progress_info["commits_processed"] > 0:
            commits_per_second = progress_info["commits_processed"] / elapsed.total_seconds()
            remaining_commits = progress_info["total_commits"] - progress_info["commits_processed"]
            if commits_per_second > 0:
                eta_seconds = remaining_commits / commits_per_second
                progress_info["eta_seconds"] = eta_seconds
            else:
                progress_info["eta_seconds"] = None
        else:
            progress_info["eta_seconds"] = None

        # Format times for human readability
        progress_info["elapsed_human"] = f"{elapsed.total_seconds():.1f}s"
        if progress_info["eta_seconds"]:
            progress_info["eta_human"] = f"{progress_info['eta_seconds']:.1f}s"
        else:
            progress_info["eta_human"] = "unknown"

        return progress_info

    async def _validate_and_repair_repository_state(self, repository_id: str, repository) -> bool:
        """Validate repository state and attempt to repair inconsistencies."""
        try:
            # Check if repository status is consistent with database state
            commit_count = await self.db.get_commit_count(repository_id)

            # If repository shows as INDEXED but has no commits, mark as not indexed
            if repository.status == RepositoryStatus.INDEXED and commit_count == 0:
                logger.warning(
                    f"Repository {repository_id} marked as INDEXED but has no commits. Resetting state."
                )
                await self.db.update_repository_status(repository_id, RepositoryStatus.NOT_INDEXED)
                return False

            # If repository shows as INDEXING but no active background task, reset to appropriate state
            if (
                repository.status == RepositoryStatus.INDEXING
                and repository_id not in self._background_tasks
            ):
                logger.warning(
                    f"Repository {repository_id} marked as INDEXING but no active task. Resetting state."
                )
                if commit_count > 0:
                    await self.db.update_repository_status(
                        repository_id, RepositoryStatus.INDEXED, commit_count=commit_count
                    )
                else:
                    await self.db.update_repository_status(
                        repository_id, RepositoryStatus.NOT_INDEXED
                    )
                return True

            # If repository shows as FAILED, clear error if we can proceed
            if repository.status == RepositoryStatus.FAILED:
                logger.info(
                    f"Repository {repository_id} was in FAILED state. Attempting to recover."
                )
                if commit_count > 0:
                    await self.db.update_repository_status(
                        repository_id, RepositoryStatus.INDEXED, commit_count=commit_count
                    )
                else:
                    await self.db.update_repository_status(
                        repository_id, RepositoryStatus.NOT_INDEXED
                    )
                return True

            return True
        except Exception as e:
            logger.error(f"Failed to validate/repair repository state for {repository_id}: {e}")
            return False

    async def _check_and_update_if_stale(self, repository_id: str, repository) -> Dict[str, Any]:
        """Check if index is stale and update appropriately.

        Returns:
            Dict with keys:
            - updated: bool - whether any update was performed
            - background: bool - whether update is running in background
            - commit_gap: int - number of commits found
            - warning_message: str - message for user if background indexing started
        """
        result = {"updated": False, "background": False, "commit_gap": 0, "warning_message": ""}

        if not self.enable_auto_update:
            return result

        is_stale, commit_gap = await self._is_index_stale(repository_id, repository.canonical_path)
        result["commit_gap"] = commit_gap

        if not is_stale:
            return result

        if commit_gap == -1:
            # No commits indexed yet - this should be handled by existing flow
            return result

        if commit_gap >= self.background_threshold:
            # Check if background task is already running for this repository
            if repository_id in self._background_tasks:
                task = self._background_tasks[repository_id]
                if not task.done():
                    # Task already running, return info about it
                    result["background"] = True
                    result["warning_message"] = (
                        f"Background indexing already in progress for {commit_gap} new commits. "
                        f"Search results may not include the latest commits until indexing completes."
                    )
                    return result
                else:
                    # Task completed, clean it up
                    del self._background_tasks[repository_id]

            # Start background indexing for large gaps
            logger.info(
                f"Starting background indexing for {commit_gap} new commits (≥{self.background_threshold})"
            )

            # Start background task with retry logic (fire and forget) and track it
            task = asyncio.create_task(
                self._start_background_indexing_with_retry(repository_id, repository)
            )
            self._background_tasks[repository_id] = task

            result["background"] = True

            # Check if we have progress information for enhanced warning
            progress_info = self.get_background_task_progress(repository_id)
            if progress_info and progress_info["commits_processed"] > 0:
                # Background task is actively running
                percent = progress_info.get("progress_percent", 0)
                processed = progress_info["commits_processed"]
                total = progress_info["total_commits"]
                eta = progress_info["eta_human"]
                result["warning_message"] = (
                    f"Background indexing in progress: {processed}/{total} commits ({percent}%) - "
                    f"ETA: {eta}. Search results may not include the latest commits."
                )
            else:
                # Just starting background indexing
                result["warning_message"] = (
                    f"Found {commit_gap} new commits. Indexing in background - "
                    f"search results may not include the latest commits until indexing completes."
                )
            return result

        # Foreground indexing for smaller gaps
        logger.info(f"Auto-updating index with {commit_gap} new commits...")

        try:
            await self.db.update_repository_status(
                repository_id, RepositoryStatus.INDEXING, progress=0
            )

            async def get_commits(git_repo):
                latest_date = await self.db.get_latest_commit_date(repository_id)
                return await git_repo.get_commits_since(latest_date)

            commits = await self._with_git_repo(repository.canonical_path, get_commits)

            if not commits:
                await self.db.update_repository_status(
                    repository_id,
                    RepositoryStatus.INDEXED,
                    commit_count=await self.db.get_commit_count(repository_id),
                )
                result["updated"] = True
                return result

            # Process new commits
            for commit_data in commits:
                if await self.db.commit_exists(repository_id, commit_data["sha"]):
                    continue

                content = self.embeddings.format_commit_for_embedding(
                    message=commit_data["message"],
                    diff=commit_data.get("diff", ""),
                    files_changed=commit_data.get("files_changed", []),
                )

                embedding = await self.embeddings.generate_embedding(
                    content, files_changed=commit_data.get("files_changed", [])
                )

                from spelungit.models import StoredCommit

                stored_commit = StoredCommit(
                    repository_id=repository_id,
                    sha=commit_data["sha"],
                    embedding=embedding,
                    commit_date=commit_data["date"],
                    created_at=datetime.now(),
                )

                authors = commit_data.get("authors", [commit_data.get("author", "")])
                await self.db.store_commit(
                    stored_commit,
                    authors,
                    message=commit_data.get("message", ""),
                    diff_content=commit_data.get("diff", ""),
                )

            # Mark as completed
            total_count = await self.db.get_commit_count(repository_id)
            await self.db.update_repository_status(
                repository_id, RepositoryStatus.INDEXED, commit_count=total_count
            )

            # Clear cache for this repository
            if repository_id in self._staleness_cache:
                del self._staleness_cache[repository_id]

            logger.info(f"✅ Auto-updated index with {len(commits)} new commits")
            result["updated"] = True
            return result

        except Exception as e:
            logger.error(f"Auto-update failed for {repository_id}: {e}")

            # Clear staleness cache to force re-evaluation on next attempt
            if repository_id in self._staleness_cache:
                del self._staleness_cache[repository_id]

            # Update repository status with error information
            await self.db.update_repository_status(
                repository_id,
                RepositoryStatus.INDEXED,  # Revert to previous status
                error_message=f"Auto-update failed: {str(e)[:200]}",
            )
            return result

    async def search_commits(
        self,
        query: str,
        repository_path: Optional[str] = None,
        limit: int = 10,
        author_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search commits using vector similarity within detected repository."""

        # Detect repository context
        repository_id, repository = await self._detect_repository_context(repository_path)

        # Validate and repair repository state if needed (with timeout)
        try:
            await asyncio.wait_for(
                self._validate_and_repair_repository_state(repository_id, repository), timeout=5.0
            )
            # Refresh repository info after potential state repair
            repository = await self.db.get_or_create_repository(
                repository_id, repository.canonical_path
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"State validation timed out for {repository_id}, proceeding with current state"
            )
        except Exception as e:
            logger.warning(
                f"State validation failed for {repository_id}: {e}, proceeding with current state"
            )

        # Check repository status
        if repository.status == RepositoryStatus.NOT_INDEXED:
            commit_count = await self._estimate_commit_count(repository.canonical_path)
            raise RepositoryNotIndexedException(
                f"Repository '{repository_id}' is not indexed. "
                f"Estimated {commit_count} commits to process. "
                f"Use the 'index_repository' tool to begin indexing."
            )
        elif repository.status == RepositoryStatus.INDEXING:
            progress = repository.indexing_progress or 0
            raise RepositoryIndexingException(
                f"Repository '{repository_id}' is being indexed ({progress}% complete). "
                f"Please wait for indexing to complete."
            )
        elif repository.status == RepositoryStatus.FAILED:
            error_msg = repository.error_message or "Unknown error"
            raise Exception(
                f"Repository '{repository_id}' indexing failed: {error_msg}. "
                f"Use the 'index_repository' tool to retry."
            )

        # Check if index is stale and auto-update if possible
        update_info = None
        if repository.status == RepositoryStatus.INDEXED:
            try:
                update_info = await self._check_and_update_if_stale(repository_id, repository)
                if update_info["updated"]:
                    # Refresh repository info after foreground update
                    repository = await self.db.get_or_create_repository(
                        repository_id, repository.canonical_path
                    )
            except Exception as e:
                # Log warning but don't fail the search
                logger.warning(f"Auto-update check failed for {repository_id}: {e}")

        # Generate query embedding
        query_embedding = await self.embeddings.generate_embedding(query)

        # Search for similar commits (with hybrid optimization)
        search_results = await self.db.search_commits(
            repository_id=repository_id,
            query_embedding=query_embedding,
            limit=limit,
            author_filter=author_filter,
            query_text=query,  # Pass original query for FTS optimization
        )

        # Get commit details from Git
        async def get_all_commit_details(git_repo):
            results = []
            for result in search_results:
                try:
                    commit_info = await git_repo.get_commit_info(result.sha)
                    results.append(
                        {
                            "sha": result.sha,
                            "similarity_score": result.similarity_score,
                            "message": commit_info.get("message", ""),
                            "author": commit_info.get("author", ""),
                            "date": commit_info.get("date", ""),
                            "files_changed": commit_info.get("files_changed", []),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Could not get details for commit {result.sha}: {e}")
                    continue
            return results

        results = await self._with_git_repo(repository.canonical_path, get_all_commit_details)

        # Include background indexing warning if applicable
        if update_info and update_info.get("warning_message"):
            # Add warning message to results metadata (will be handled by tool call handler)
            results.append(
                {
                    "_warning": update_info["warning_message"],
                    "_background_indexing": True,
                    "_commit_gap": update_info.get("commit_gap", 0),
                }
            )

        return results

    async def _estimate_commit_count(self, repository_path: str) -> int:
        """Estimate number of commits in repository."""

        async def get_count(git_repo):
            return await git_repo.get_commit_count()

        try:
            return await self._with_git_repo(repository_path, get_count)
        except Exception:
            return 0

    async def index_repository(
        self, repository_path: Optional[str] = None, batch_size: int = 100
    ) -> Dict[str, Any]:
        """Index a repository for search."""

        # Detect repository context
        repository_id, repository = await self._detect_repository_context(repository_path)

        logger.info(f"Starting indexing for repository: {repository_id}")
        await self.db.update_repository_status(repository_id, RepositoryStatus.INDEXING, progress=0)

        try:
            git_repo = GitRepository(repository.canonical_path)

            # Get commits that need indexing
            latest_date = await self.db.get_latest_commit_date(repository_id)
            commits = await git_repo.get_commits_since(latest_date)

            total_commits = len(commits)
            logger.info(f"Found {total_commits} commits to index")

            if total_commits == 0:
                await self.db.update_repository_status(
                    repository_id,
                    RepositoryStatus.INDEXED,
                    commit_count=await self.db.get_commit_count(repository_id),
                )
                return {
                    "repository_id": repository_id,
                    "status": "up_to_date",
                    "commits_processed": 0,
                    "total_commits": await self.db.get_commit_count(repository_id),
                }

            processed = 0
            for i in range(0, total_commits, batch_size):
                batch = commits[i : i + batch_size]

                for commit_data in batch:
                    # Check if already exists
                    if await self.db.commit_exists(repository_id, commit_data["sha"]):
                        continue

                    # Format content for embedding
                    content = self.embeddings.format_commit_for_embedding(
                        message=commit_data["message"],
                        diff=commit_data.get("diff", ""),
                        files_changed=commit_data.get("files_changed", []),
                    )

                    # Generate embedding
                    embedding = await self.embeddings.generate_embedding(
                        content, files_changed=commit_data.get("files_changed", [])
                    )

                    # Store in database
                    from spelungit.models import StoredCommit

                    stored_commit = StoredCommit(
                        repository_id=repository_id,
                        sha=commit_data["sha"],
                        embedding=embedding,
                        commit_date=commit_data["date"],
                        created_at=datetime.now(),
                    )

                    authors = commit_data.get("authors", [commit_data.get("author", "")])
                    await self.db.store_commit(
                        stored_commit,
                        authors,
                        message=commit_data.get("message", ""),
                        diff_content=commit_data.get("diff", ""),
                    )

                    processed += 1

                # Update progress
                progress = int((processed / total_commits) * 100)
                await self.db.update_repository_status(
                    repository_id, RepositoryStatus.INDEXING, progress=progress
                )

                logger.info(f"Indexed {processed}/{total_commits} commits ({progress}%)")

            # Mark as completed
            total_count = await self.db.get_commit_count(repository_id)
            await self.db.update_repository_status(
                repository_id, RepositoryStatus.INDEXED, commit_count=total_count
            )

            logger.info(f"✅ Repository indexing completed: {repository_id}")

            return {
                "repository_id": repository_id,
                "status": "completed",
                "commits_processed": processed,
                "total_commits": total_count,
            }

        except Exception as e:
            logger.error(f"Indexing failed for {repository_id}: {e}")
            await self.db.update_repository_status(
                repository_id, RepositoryStatus.FAILED, error_message=str(e)
            )
            raise


# Initialize MCP server
server = Server("git-history-mcp-lite")


@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available resources."""
    return [
        Resource(
            uri=AnyUrl("git://current-repository"),
            name="Current Repository",
            description="Information about the current Git repository",
            mimeType="application/json",
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a specific resource."""
    if uri == "git://current-repository":
        try:
            repo_info = get_repository_info(os.getcwd())
            return str(repo_info)
        except Exception as e:
            return f"Error reading repository info: {e}"

    raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_commits",
            description="Search Git commits using semantic similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "repository_path": {
                        "type": "string",
                        "description": "Optional path to repository (defaults to current directory)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10,
                    },
                    "author_filter": {
                        "type": "string",
                        "description": "Optional filter by author name",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="index_repository",
            description="Index a Git repository for semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "repository_path": {
                        "type": "string",
                        "description": "Optional path to repository (defaults to current directory)",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Number of commits to process in each batch (default: 100)",
                        "default": 100,
                    },
                },
            },
        ),
        Tool(
            name="repository_status",
            description="Get the indexing status of a repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "repository_path": {
                        "type": "string",
                        "description": "Optional path to repository (defaults to current directory)",
                    }
                },
            },
        ),
        Tool(
            name="get_database_info",
            description="Get information about the database and indexed repositories",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="search_blame",
            description="Search code blame data using natural language",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "file_path": {
                        "type": "string",
                        "description": "Optional specific file to search blame in",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="who_wrote",
            description="Find authors who wrote code matching a query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of authors to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="configure_auto_update",
            description="Configure automatic index update behavior",
            inputSchema={
                "type": "object",
                "properties": {
                    "enable_auto_update": {
                        "type": "boolean",
                        "description": "Enable or disable automatic index updates",
                    },
                    "background_threshold": {
                        "type": "integer",
                        "description": "Threshold for background vs foreground indexing (default: 50)",
                        "minimum": 5,
                        "maximum": 500,
                    },
                    "staleness_check_cache_minutes": {
                        "type": "integer",
                        "description": "Cache validity for staleness checks in minutes (default: 5)",
                        "minimum": 1,
                        "maximum": 60,
                    },
                },
            },
        ),
        Tool(
            name="get_auto_update_config",
            description="Get current automatic index update configuration",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    global db_manager, embedding_manager

    # Initialize if not already done
    if not db_manager:
        db_manager = SQLiteDatabaseManager()
        await db_manager.initialize()

    if not embedding_manager:
        embedding_manager = LiteEmbeddingManager()

    search_engine = LiteSearchEngine(db_manager, embedding_manager)

    try:
        if name == "search_commits":
            query = arguments["query"]
            repository_path = arguments.get("repository_path")
            limit = arguments.get("limit", 10)
            author_filter = arguments.get("author_filter")

            try:
                results = await search_engine.search_commits(
                    query=query,
                    repository_path=repository_path,
                    limit=limit,
                    author_filter=author_filter,
                )

                # Check for warning messages (background indexing)
                warning_message = ""
                actual_results = []
                for result in results:
                    if isinstance(result, dict) and "_warning" in result:
                        warning_message = result["_warning"]
                    else:
                        actual_results.append(result)

                if not actual_results:
                    message = "No matching commits found."
                    if warning_message:
                        message += f"\n\n⚠️  {warning_message}"
                    return [TextContent(type="text", text=message)]

                # Format results
                formatted_results = []
                for result in actual_results:
                    formatted_results.append(
                        f"**Commit:** {result['sha'][:8]}\n"
                        f"**Similarity:** {result['similarity_score']:.3f}\n"
                        f"**Author:** {result['author']}\n"
                        f"**Date:** {result['date']}\n"
                        f"**Message:** {result['message']}\n"
                        f"**Files:** {', '.join(result['files_changed'][:5])}"
                        + (
                            f" (+{len(result['files_changed']) - 5} more)"
                            if len(result["files_changed"]) > 5
                            else ""
                        )
                    )

                response_text = (
                    f"Found {len(actual_results)} matching commits:\n\n"
                    + "\n\n---\n\n".join(formatted_results)
                )

                # Add warning message if background indexing is happening
                if warning_message:
                    response_text += f"\n\n⚠️  {warning_message}"

                return [TextContent(type="text", text=response_text)]

            except (RepositoryNotIndexedException, RepositoryIndexingException) as e:
                return [TextContent(type="text", text=str(e))]

        elif name == "index_repository":
            repository_path = arguments.get("repository_path")
            batch_size = arguments.get("batch_size", 100)

            result = await search_engine.index_repository(
                repository_path=repository_path, batch_size=batch_size
            )

            status_text = (
                f"**Repository:** {result['repository_id']}\n"
                f"**Status:** {result['status']}\n"
                f"**Commits Processed:** {result['commits_processed']}\n"
                f"**Total Commits:** {result['total_commits']}"
            )

            return [TextContent(type="text", text=status_text)]

        elif name == "repository_status":
            repository_path = arguments.get("repository_path")

            try:
                repository_id, repository = await search_engine._detect_repository_context(
                    repository_path
                )

                status_text = (
                    f"**Repository:** {repository_id}\n"
                    f"**Path:** {repository.canonical_path}\n"
                    f"**Status:** {repository.status.value}\n"
                    f"**Commits:** {repository.commit_count or 0}\n"
                    f"**Last Indexed:** {repository.last_indexed or 'Never'}"
                )

                if repository.status == RepositoryStatus.INDEXING:
                    status_text += f"\n**Progress:** {repository.indexing_progress or 0}%"

                if repository.error_message:
                    status_text += f"\n**Error:** {repository.error_message}"

                # Add auto-update information
                if repository.status == RepositoryStatus.INDEXED:
                    try:
                        is_stale, commit_gap = await search_engine._is_index_stale(
                            repository_id, repository.canonical_path
                        )
                        if is_stale and commit_gap > 0:
                            status_text += (
                                f"\n**Index Status:** Stale ({commit_gap} new commits available)"
                            )
                            if commit_gap < search_engine.background_threshold:
                                status_text += (
                                    "\n**Auto-update:** Will update immediately on next search"
                                )
                            else:
                                status_text += (
                                    "\n**Auto-update:** Will update in background on next search"
                                )
                        else:
                            status_text += "\n**Index Status:** Up to date"
                    except Exception as e:
                        logger.warning(f"Could not check staleness for status: {e}")

                # Add background task progress if available
                progress_info = search_engine.get_background_task_progress(repository_id)
                if progress_info:
                    status_text += "\n\n**Background Indexing:**"
                    status_text += f"\n• Phase: {progress_info['phase']}"
                    status_text += f"\n• Progress: {progress_info['commits_processed']}/{progress_info['total_commits']} commits"
                    if progress_info.get("progress_percent"):
                        status_text += f" ({progress_info['progress_percent']}%)"
                    status_text += f"\n• Batch: {progress_info['current_batch']}/{progress_info['total_batches']}"
                    status_text += f"\n• Elapsed: {progress_info['elapsed_human']}"
                    status_text += f"\n• ETA: {progress_info['eta_human']}"

                # Add auto-update configuration info
                status_text += (
                    f"\n\n**Auto-update Config:**\n"
                    f"• Enabled: {'Yes' if search_engine.enable_auto_update else 'No'}\n"
                    f"• Background Threshold: {search_engine.background_threshold} commits\n"
                    f"• Cache: {search_engine.staleness_check_cache_minutes} min"
                )

                return [TextContent(type="text", text=status_text)]

            except Exception as e:
                return [TextContent(type="text", text=f"Error: {e}")]

        elif name == "get_database_info":
            db_info = await db_manager.get_database_info()

            info_text = (
                f"**Database Type:** {db_info['database_type']}\n"
                f"**Database Path:** {db_info['database_path']}\n"
                f"**Database Size:** {db_info['database_size_mb']} MB\n"
                f"**Repositories:** {db_info['repositories']}\n"
                f"**Total Commits:** {db_info['total_commits']}\n"
                f"**Embedding Model:** {embedding_manager.model_info}\n"
                f"**Vector Search:** ✅\n"
                f"**Full-text Search:** ✅"
            )

            return [TextContent(type="text", text=info_text)]

        elif name == "search_blame":
            query = arguments["query"]
            file_path = arguments.get("file_path")
            limit = arguments.get("limit", 10)

            try:
                # Use existing search to find relevant commits first
                repository_id, repository = await search_engine._detect_repository_context()
                search_results = await search_engine.search_commits(query=query, limit=20)

                if not search_results:
                    return [
                        TextContent(
                            type="text", text="No matching commits found for blame analysis"
                        )
                    ]

                # Initialize git repository
                git_repo = GitRepository(repository.canonical_path)
                blame_results = []
                files_processed = set()

                for result in search_results[:5]:  # Limit to first 5 commits
                    try:
                        if file_path:
                            target_files = [file_path]
                        else:
                            # Get files changed in this commit
                            commit_info = await git_repo.get_commit_info(result["sha"])
                            target_files = commit_info.get("files_changed", [])[:3]

                        for target_file in target_files:
                            if target_file in files_processed:
                                continue
                            files_processed.add(target_file)

                            blame_info = await git_repo.get_blame_info(target_file)
                            if blame_info:
                                # Simple relevance filtering
                                query_words = query.lower().split()
                                relevant_lines = [
                                    entry
                                    for entry in blame_info
                                    if any(word in entry["content"].lower() for word in query_words)
                                ]

                                if relevant_lines:
                                    blame_results.append(
                                        {
                                            "file_path": target_file,
                                            "relevant_lines": relevant_lines[:10],
                                            "total_lines": len(blame_info),
                                        }
                                    )

                    except Exception as e:
                        logger.warning(f"Error processing blame for commit {result['sha']}: {e}")
                        continue

                # Format results
                if not blame_results:
                    return [
                        TextContent(
                            type="text",
                            text=f"No blame information found matching '{query}'. Try with a specific file path.",
                        )
                    ]

                response_parts = [f"**Blame Analysis for '{query}'**\n"]
                for file_result in blame_results[:3]:
                    response_parts.append(f"\n**File:** `{file_result['file_path']}`")
                    response_parts.append(
                        f"**Relevant Lines:** {len(file_result['relevant_lines'])} of {file_result['total_lines']} total\n"
                    )

                    for line_info in file_result["relevant_lines"][:5]:
                        response_parts.append(
                            f"**Line {line_info['line_number']}:** {line_info['author']} "
                            f"({line_info['commit_sha'][:8]}, {line_info['date'][:10]})"
                        )
                        response_parts.append(f"```\n{line_info['content']}\n```")
                        if line_info["summary"]:
                            response_parts.append(f"*Commit: {line_info['summary']}*\n")

                return [TextContent(type="text", text="\n".join(response_parts))]

            except Exception as e:
                logger.error(f"Error in search_blame: {e}")
                return [TextContent(type="text", text=f"Error searching blame: {str(e)}")]

        elif name == "who_wrote":
            query = arguments["query"]
            limit = arguments.get("limit", 5)

            try:
                # Detect repository context
                repository_id, repository = await search_engine._detect_repository_context()

                # Generate query embedding
                query_embedding = await embedding_manager.generate_embedding(query)

                # Get authors who wrote relevant code
                authors = await db_manager.get_authors_for_query(
                    repository_id=repository_id,
                    query_embedding=query_embedding,
                    query_text=query,
                    limit=limit,
                )

                if not authors:
                    return [
                        TextContent(
                            type="text",
                            text=f"No authors found who wrote code matching '{query}'. Ensure the repository is indexed.",
                        )
                    ]

                # Format results
                response_parts = [f"**Authors who wrote code matching '{query}':**\n"]

                for i, author_data in enumerate(authors, 1):
                    author_name = author_data["author"]
                    commit_count = author_data["commit_count"]
                    max_relevance = author_data["max_relevance"]
                    avg_relevance = author_data["avg_relevance"]
                    latest_date = author_data["latest_contribution"][:10]

                    response_parts.append(
                        f"**{i}. {author_name}** ({commit_count} matching commits)\n"
                        f"   **Relevance:** Max: {max_relevance:.3f}, Avg: {avg_relevance:.3f}\n"
                        f"   **Latest:** {latest_date}\n"
                    )

                    # Show example commits
                    if author_data["example_commits"]:
                        response_parts.append("   **Example commits:**")
                        for commit in author_data["example_commits"][:2]:
                            response_parts.append(
                                f"   • `{commit['sha']}` {commit['message'][:60]}{'...' if len(commit['message']) > 60 else ''} "
                                f"(relevance: {commit['relevance']:.3f})"
                            )
                        response_parts.append("")

                return [TextContent(type="text", text="\n".join(response_parts))]

            except Exception as e:
                logger.error(f"Error in who_wrote: {e}")
                return [TextContent(type="text", text=f"Error finding authors: {str(e)}")]

        elif name == "configure_auto_update":
            enable_auto_update = arguments.get("enable_auto_update")
            background_threshold = arguments.get("background_threshold")
            staleness_check_cache_minutes = arguments.get("staleness_check_cache_minutes")

            try:
                # Update configuration in the search engine
                config_updated = []

                if enable_auto_update is not None:
                    search_engine.enable_auto_update = enable_auto_update
                    config_updated.append(
                        f"Auto-update: {'enabled' if enable_auto_update else 'disabled'}"
                    )

                if background_threshold is not None:
                    search_engine.background_threshold = background_threshold
                    config_updated.append(f"Background threshold: {background_threshold} commits")

                if staleness_check_cache_minutes is not None:
                    search_engine.staleness_check_cache_minutes = staleness_check_cache_minutes
                    config_updated.append(
                        f"Cache duration: {staleness_check_cache_minutes} minutes"
                    )
                    # Clear existing cache since cache duration changed
                    search_engine._staleness_cache.clear()

                if config_updated:
                    response_text = "**Auto-update configuration updated:**\n\n" + "\n".join(
                        f"• {item}" for item in config_updated
                    )
                else:
                    response_text = "No configuration changes specified."

                return [TextContent(type="text", text=response_text)]

            except Exception as e:
                logger.error(f"Error configuring auto-update: {e}")
                return [TextContent(type="text", text=f"Error configuring auto-update: {str(e)}")]

        elif name == "get_auto_update_config":
            try:
                config_text = (
                    "**Current Auto-update Configuration:**\n\n"
                    f"**Enabled:** {'Yes' if search_engine.enable_auto_update else 'No'}\n"
                    f"**Background Threshold:** {search_engine.background_threshold} commits\n"
                    f"**Cache Duration:** {search_engine.staleness_check_cache_minutes} minutes\n\n"
                    f"**Behavior:**\n"
                    f"• < {search_engine.background_threshold} commits: Immediate foreground indexing\n"
                    f"• ≥ {search_engine.background_threshold} commits: Background indexing with warning\n\n"
                    f"**Cache Status:** {len(search_engine._staleness_cache)} repositories cached"
                )

                return [TextContent(type="text", text=config_text)]

            except Exception as e:
                logger.error(f"Error getting auto-update config: {e}")
                return [TextContent(type="text", text=f"Error getting configuration: {str(e)}")]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


async def main():
    """Main entry point for the MCP server."""
    if not HAS_MCP:
        print("❌ MCP library not available. Please install with: pip install mcp")
        sys.exit(1)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize database and embeddings
    global db_manager, embedding_manager

    try:
        db_manager = SQLiteDatabaseManager()
        await db_manager.initialize()
        logger.info("✅ SQLite database initialized")

        embedding_manager = LiteEmbeddingManager()
        logger.info(f"✅ Embedding manager initialized: {embedding_manager.model_info}")

    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        sys.exit(1)

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="git-history-mcp-lite",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


async def test_lite_search():
    """Test the lite search functionality without MCP server."""
    print("🧪 Testing Lite Search Engine")
    print("=" * 40)

    # Initialize components
    db_manager = SQLiteDatabaseManager()
    await db_manager.initialize()

    embedding_manager = LiteEmbeddingManager()
    search_engine = LiteSearchEngine(db_manager, embedding_manager)

    print(f"✅ Database initialized: {db_manager.db_path}")
    print(f"✅ Embedding model: {embedding_manager.model_info}")

    # Test repository detection
    try:
        repo_id, repo = await search_engine._detect_repository_context()
        print(f"✅ Repository detected: {repo_id}")
        print(f"   Path: {repo.canonical_path}")
        print(f"   Status: {repo.status.value}")
    except Exception as e:
        print(f"❌ Repository detection failed: {e}")

    await db_manager.close()
    print("\n✅ Lite search engine test completed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_lite_search())
    elif HAS_MCP:
        asyncio.run(main())
    else:
        print("❌ This module requires the MCP library. Install with:")
        print("pip install mcp")
        print("\nOr use the installation script: ./install.sh")
        print("\nTo test without MCP, use: python -m spelungit.lite_server --test")
        sys.exit(1)
