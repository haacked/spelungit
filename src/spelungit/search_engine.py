"""
Semantic search engine for Git commit history.
Handles vector similarity search, ranking, and result formatting.
Supports multi-repository operation with Git worktree detection.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Repository,
    RepositoryIndexingException,
    RepositoryNotIndexedException,
    RepositoryStatus,
    SearchResult,
)
from .repository_utils import detect_repository_context

logger = logging.getLogger(__name__)


class MockSearchDatabase:
    """Mock database that supports vector similarity search with repository support."""

    def __init__(self):
        self.repositories: Dict[str, Repository] = {}  # repository_id -> Repository object
        self.commits: Dict[
            Tuple[str, str], Dict[str, Any]
        ] = {}  # (repository_id, sha) -> {embedding, metadata}
        self.authors: Dict[Tuple[str, str], List[str]] = {}  # (repository_id, sha) -> [authors]

    async def get_or_create_repository(self, repository_id: str, canonical_path: str) -> Repository:
        """Get existing repository or create a new one."""
        if repository_id not in self.repositories:
            self.repositories[repository_id] = Repository(
                repository_id=repository_id,
                canonical_path=canonical_path,
                discovered_paths=[canonical_path],
                status=RepositoryStatus.NOT_INDEXED,
                created_at=datetime.now(),
            )
        return self.repositories[repository_id]

    async def update_repository_status(
        self,
        repository_id: str,
        status: RepositoryStatus,
        commit_count: Optional[int] = None,
        progress: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update repository status and metadata."""
        if repository_id in self.repositories:
            repo = self.repositories[repository_id]
            repo.status = status
            if commit_count is not None:
                repo.commit_count = commit_count
            if progress is not None:
                repo.indexing_progress = progress
            if error_message is not None:
                repo.error_message = error_message
            if status == RepositoryStatus.INDEXED:
                repo.last_indexed = datetime.now()

    async def store_commit_with_embedding(
        self,
        repository_id: str,
        sha: str,
        embedding: List[float],
        metadata: Dict,
        authors: List[str],
    ):
        """Store a commit with its embedding and metadata."""
        key = (repository_id, sha)
        self.commits[key] = {"embedding": embedding, "metadata": metadata}
        self.authors[key] = authors

    async def search_by_embedding(
        self,
        repository_id: str,
        query_embedding: List[float],
        limit: int = 10,
        author_filter: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[SearchResult]:
        """Search for similar commits using vector similarity within a repository."""
        results = []

        for key, commit_data in self.commits.items():
            repo_id, sha = key

            # Only search within the specified repository
            if repo_id != repository_id:
                continue

            # Apply filters
            if author_filter:
                commit_authors = self.authors.get(key, [])
                if not any(author_filter.lower() in author.lower() for author in commit_authors):
                    continue

            if date_range:
                commit_date = commit_data["metadata"].get("commit_date")
                if commit_date:
                    # Ensure all datetimes are timezone-aware for comparison
                    if commit_date.tzinfo is None:
                        commit_date = commit_date.replace(tzinfo=datetime.now().astimezone().tzinfo)
                    start_date, end_date = date_range
                    if start_date.tzinfo is None:
                        start_date = start_date.replace(tzinfo=datetime.now().astimezone().tzinfo)
                    if end_date.tzinfo is None:
                        end_date = end_date.replace(tzinfo=datetime.now().astimezone().tzinfo)

                    if not (start_date <= commit_date <= end_date):
                        continue

            # Calculate similarity
            similarity = self._calculate_cosine_similarity(
                query_embedding, commit_data["embedding"]
            )

            result = SearchResult(repository_id=repository_id, sha=sha, similarity_score=similarity)
            results.append(result)

        # Sort by similarity score (descending)
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        return results[:limit]

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return float(dot_product / (magnitude1 * magnitude2))

    async def get_commit_count(self, repository_id: Optional[str] = None) -> int:
        if repository_id:
            return sum(1 for key in self.commits.keys() if key[0] == repository_id)
        return len(self.commits)


class SearchEngine:
    """Main search engine that orchestrates semantic search across Git history with repository auto-detection."""

    def __init__(self, database, git_scanner, embedding_manager):
        self.db = database
        self.git = git_scanner
        self.embeddings = embedding_manager

    async def search_commits(
        self,
        query: str,
        limit: int = 10,
        author_filter: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        include_details: bool = True,
        repository_path: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search commits using natural language query with repository auto-detection.

        Args:
            query: Natural language search query
            limit: Maximum number of results
            author_filter: Filter by author name/email
            date_range: Tuple of (start_date, end_date)
            include_details: Whether to fetch full commit details
            repository_path: Optional repository path (auto-detects if None)

        Returns:
            List of SearchResult objects with similarity scores

        Raises:
            RepositoryNotIndexedException: If repository is not indexed
            RepositoryIndexingException: If repository is currently being indexed
        """
        logger.info(f"Searching for: '{query}' (limit={limit})")

        # Handle empty query
        if not query.strip():
            logger.warning("Empty query provided, using default search")
            query = "recent changes"

        # Detect repository context
        repository_id, repository = await detect_repository_context(self.db, repository_path)
        logger.info(f"Repository detected: {repository_id} ({repository.status.value})")

        # Check indexing status and handle appropriately
        if repository.status == RepositoryStatus.NOT_INDEXED:
            raise RepositoryNotIndexedException(
                f"Repository '{repository_id}' is not indexed. Indexing will start automatically."
            )
        elif repository.status == RepositoryStatus.INDEXING:
            progress = repository.indexing_progress or 0
            raise RepositoryIndexingException(
                f"Repository '{repository_id}' is being indexed ({progress}% complete). "
                f"Please try again in a few minutes."
            )
        elif repository.status == RepositoryStatus.FAILED:
            error_msg = repository.error_message or "Unknown error"
            raise RepositoryIndexingException(
                f"Repository '{repository_id}' indexing failed: {error_msg}. "
                f"Please check logs or try re-indexing."
            )

        # Generate embedding for search query
        query_embedding = await self.embeddings.generate_embedding(query)

        # Search database for similar commits within the repository
        search_results = await self.db.search_by_embedding(
            repository_id=repository_id,
            query_embedding=query_embedding,
            limit=limit * 2,  # Get more to allow for filtering
            author_filter=author_filter,
            date_range=date_range,
        )

        logger.info(f"Found {len(search_results)} initial matches")

        # Fetch commit details if requested
        if include_details and search_results:
            await self._enrich_with_commit_details(search_results, repository_id)

        # Apply additional ranking and filtering
        ranked_results = self._rank_and_filter_results(search_results, query)

        # Apply final limit
        final_results = ranked_results[:limit]

        # Set final ranks
        for i, result in enumerate(final_results):
            result.rank = i + 1

        logger.info(f"Returning {len(final_results)} results")
        return final_results

    async def _enrich_with_commit_details(
        self, search_results: List[SearchResult], repository_id: str
    ):
        """Fetch full commit details for search results."""
        shas = [result.sha for result in search_results]

        # Batch fetch commit details
        commit_details = self.git.get_commits_batch_details(shas, batch_size=20)

        # Create lookup dict
        details_by_sha = {commit["sha"]: commit for commit in commit_details}

        # Enrich results
        for result in search_results:
            result.commit_info = details_by_sha.get(result.sha)

    def _rank_and_filter_results(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """Apply additional ranking based on query relevance."""
        query_words = set(query.lower().split())

        for result in results:
            # Base score is similarity
            score = result.similarity_score

            if result.commit_info:
                # Boost if query terms appear in commit message
                message_words = set(result.commit_info.message.lower().split())
                word_overlap = len(query_words & message_words)
                if word_overlap > 0:
                    score += 0.1 * word_overlap

                # Boost recent commits slightly
                commit_age_days = (
                    datetime.now() - result.commit_info.commit_date.replace(tzinfo=None)
                ).days
                if commit_age_days < 30:  # Recent commits
                    score += 0.05
                elif commit_age_days < 365:  # This year
                    score += 0.02

                # Boost commits with co-authors (collaborative work)
                if result.commit_info.co_authors:
                    score += 0.03

            # Update similarity score with ranking adjustments
            result.similarity_score = min(1.0, score)  # Cap at 1.0

        # Re-sort by adjusted score
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results

    def format_search_results(self, results: List[SearchResult], query: str) -> str:
        """Format search results for display."""
        if not results:
            return f"No commits found matching '{query}'."

        response_lines = [f"Found {len(results)} commits matching '{query}':\n"]

        for result in results:
            info = result.commit_info
            if not info:
                response_lines.append(
                    f"{result.rank}. {result.sha[:8]} (Score: {result.similarity_score:.3f})"
                )
                continue

            # Format result
            lines = [
                f"## {result.rank}. {info.sha[:8]} (Score: {result.similarity_score:.3f})",
                f"**Author:** {info.author_name} <{info.author_email}>",
                f"**Date:** {info.commit_date.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Subject:** {info.message}",
            ]

            if info.co_authors:
                lines.append(f"**Co-authors:** {', '.join(info.co_authors)}")

            lines.append(f"**Files changed:** {len(info.files_changed)}")

            # Show diff preview if available
            if info.diff:
                diff_preview = info.diff[:500]
                if len(info.diff) > 500:
                    diff_preview += "..."
                lines.append(f"**Diff preview:**\n```diff\n{diff_preview}\n```")

            lines.append("")  # Spacing
            response_lines.extend(lines)

        return "\n".join(response_lines)

    async def search_by_author(
        self, author_query: str, limit: int = 10, repository_path: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for commits by a specific author."""
        logger.info(f"Searching for commits by author: '{author_query}'")

        # This would use database author filtering
        # For now, we'll simulate with a general search but focus on author filtering
        return await self.search_commits(
            query=f"commits by {author_query}",
            limit=limit,
            author_filter=author_query,
            include_details=True,
            repository_path=repository_path,
        )

    async def search_code_changes(
        self, code_query: str, limit: int = 10, repository_path: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for specific code changes."""
        logger.info(f"Searching for code changes: '{code_query}'")

        # Generate embedding with focus on code changes
        enhanced_query = f"code changes diff: {code_query}"
        return await self.search_commits(
            query=enhanced_query, limit=limit, include_details=True, repository_path=repository_path
        )

    async def get_search_stats(self, repository_path: Optional[str] = None) -> Dict:
        """Get statistics about the search index for a repository."""
        try:
            # Detect repository context if path provided
            if repository_path:
                repository_id, repository = await detect_repository_context(
                    self.db, repository_path
                )
                commit_count = await self.db.get_commit_count(repository_id)

                return {
                    "repository_id": repository_id,
                    "canonical_path": repository.canonical_path,
                    "status": repository.status.value,
                    "indexed_commits": commit_count,
                    "last_indexed": repository.last_indexed.isoformat()
                    if repository.last_indexed
                    else None,
                    "search_engine": "Vector Similarity (Cosine)",
                    "embedding_model": getattr(self.embeddings, "model", "mock"),
                    "features": [
                        "Multi-repository support",
                        "Git worktree detection",
                        "Semantic search",
                        "Author filtering",
                        "Date range filtering",
                        "Result ranking",
                        "Batch processing",
                        "Auto-indexing",
                    ],
                }
            else:
                # Global stats across all repositories
                total_commits = await self.db.get_commit_count()

                return {
                    "total_indexed_commits": total_commits,
                    "search_engine": "Vector Similarity (Cosine)",
                    "embedding_model": getattr(self.embeddings, "model", "mock"),
                    "features": [
                        "Multi-repository support",
                        "Git worktree detection",
                        "Semantic search",
                        "Author filtering",
                        "Date range filtering",
                        "Result ranking",
                        "Batch processing",
                        "Auto-indexing",
                    ],
                }
        except Exception as e:
            logger.error(f"Error getting search stats: {e}")
            return {"error": str(e)}
