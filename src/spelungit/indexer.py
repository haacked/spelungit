"""Indexing operations for processing Git commits."""

import asyncio
import logging
from datetime import datetime
from typing import List

from .database import DatabaseManager
from .embeddings import EmbeddingManager
from .git_operations import GitManager
from .models import StoredCommit

logger = logging.getLogger(__name__)


class CommitIndexer:
    """Handles indexing of Git commits into the database."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        git_manager: GitManager,
        embedding_manager: EmbeddingManager,
        repository_id: str,
    ):
        self.db = db_manager
        self.git = git_manager
        self.embeddings = embedding_manager
        self.repository_id = repository_id

    async def index_all_commits(self, batch_size: int = 100) -> None:
        """Index all commits in the repository."""
        # Get all commit SHAs
        all_shas = await self.git.get_all_commit_shas()
        logger.info(f"Found {len(all_shas)} total commits to process")

        # Filter out already indexed commits
        unindexed_shas = []
        for sha in all_shas:
            if not await self.db.commit_exists(self.repository_id, sha):
                unindexed_shas.append(sha)

        logger.info(f"Found {len(unindexed_shas)} unindexed commits")

        if not unindexed_shas:
            logger.info("All commits are already indexed")
            return

        await self.index_commits(unindexed_shas, batch_size)

    async def index_sample_commits(self, sample_size: int, batch_size: int = 50) -> None:
        """Index a sample of recent commits for testing."""
        # Get recent commits
        all_shas = await self.git.get_all_commit_shas()

        # Take the most recent commits
        sample_shas = all_shas[:sample_size]
        logger.info(f"Indexing sample of {len(sample_shas)} recent commits")

        # Filter out already indexed
        unindexed_shas = []
        for sha in sample_shas:
            if not await self.db.commit_exists(self.repository_id, sha):
                unindexed_shas.append(sha)

        logger.info(f"Found {len(unindexed_shas)} unindexed commits in sample")

        if unindexed_shas:
            await self.index_commits(unindexed_shas, batch_size)

    async def index_commits(self, shas: List[str], batch_size: int = 100) -> None:
        """Index a specific list of commit SHAs."""
        if not shas:
            return

        total = len(shas)
        processed = 0

        logger.info(f"Starting to index {total} commits in batches of {batch_size}")

        # Process in batches
        for i in range(0, total, batch_size):
            batch_shas = shas[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_shas)} commits)")

            try:
                await self._process_commit_batch(batch_shas)
                processed += len(batch_shas)

                progress = (processed / total) * 100
                logger.info(f"Progress: {processed}/{total} commits ({progress:.1f}%)")

                # Small delay between batches to be nice to APIs
                if i + batch_size < total:
                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Continue with next batch rather than failing completely
                continue

        logger.info(f"Indexing complete. Processed {processed}/{total} commits")

    async def _process_commit_batch(self, shas: List[str]) -> None:
        """Process a single batch of commits."""
        # Step 1: Fetch commit info from Git
        commit_infos = await self.git.get_commit_batch_info(shas)

        if not commit_infos:
            logger.warning(f"No commit info fetched for batch of {len(shas)} SHAs")
            return

        logger.debug(f"Fetched info for {len(commit_infos)} commits")

        # Step 2: Generate embeddings for all commits
        embedding_texts = []
        for commit_info in commit_infos:
            text = self.embeddings.format_commit_for_embedding(
                commit_info.message, commit_info.diff
            )
            embedding_texts.append(text)

        logger.debug(f"Generating embeddings for {len(embedding_texts)} commits")
        embeddings = await self.embeddings.generate_embeddings_batch(embedding_texts)

        if len(embeddings) != len(commit_infos):
            logger.warning(f"Embedding count mismatch: {len(embeddings)} != {len(commit_infos)}")
            # Pad with zero vectors if needed
            while len(embeddings) < len(commit_infos):
                embeddings.append([0.0] * 1536)

        # Step 3: Store in database
        for commit_info, embedding in zip(commit_infos, embeddings):
            try:
                # Create stored commit record
                stored_commit = StoredCommit(
                    repository_id=self.repository_id,
                    sha=commit_info.sha,
                    embedding=embedding,
                    commit_date=commit_info.commit_date,
                    created_at=datetime.utcnow(),
                )

                # Prepare authors list (main author + co-authors)
                authors = [f"{commit_info.author_name} <{commit_info.author_email}>"]
                authors.extend(commit_info.co_authors)

                # Store in database
                await self.db.store_commit(stored_commit, authors)
                logger.debug(f"Stored commit {commit_info.sha[:8]}")

            except Exception as e:
                logger.error(f"Error storing commit {commit_info.sha}: {e}")
                continue

    async def get_indexing_progress(self) -> dict:
        """Get current indexing progress information."""
        try:
            # Get repository stats
            repo_stats = await self.git.get_repository_stats()

            # Get database stats
            indexed_count = await self.db.get_commit_count(self.repository_id)
            latest_indexed = await self.db.get_latest_commit_date(self.repository_id)

            total_commits = repo_stats["total_commits"]
            progress_percent = (indexed_count / total_commits * 100) if total_commits > 0 else 0

            return {
                "total_commits": total_commits,
                "indexed_commits": indexed_count,
                "progress_percent": progress_percent,
                "latest_indexed_date": latest_indexed,
                "repository_latest_date": repo_stats["latest_commit_date"],
                "is_up_to_date": indexed_count >= total_commits,
            }

        except Exception as e:
            logger.error(f"Error getting indexing progress: {e}")
            return {"error": str(e)}
