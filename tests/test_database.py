"""Tests for SQLite database functionality."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

from spelungit.models import Repository, RepositoryStatus, StoredCommit
from spelungit.sqlite_database import SQLiteDatabaseManager


class TestSQLiteDatabaseManager:
    """Test SQLite database manager."""

    @pytest_asyncio.fixture
    async def db_manager(self):
        """Create temporary database manager."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            db_path = temp_file.name

        manager = SQLiteDatabaseManager(db_path)
        await manager.initialize()

        yield manager

        await manager.close()
        # Clean up
        Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_database_initialization(self, db_manager):
        """Test database initialization."""
        # Should initialize without errors
        assert db_manager.conn is not None

        # Test database info
        info = await db_manager.get_database_info()
        assert info["database_type"] == "SQLite"
        assert info["repositories"] == 0
        assert info["total_commits"] == 0

    @pytest.mark.asyncio
    async def test_repository_creation(self, db_manager):
        """Test repository creation."""
        repo_id = "test-repo"
        canonical_path = "/path/to/repo"

        repository = await db_manager.get_or_create_repository(repo_id, canonical_path)

        assert isinstance(repository, Repository)
        assert repository.repository_id == repo_id
        assert repository.canonical_path == canonical_path
        assert repository.status == RepositoryStatus.NOT_INDEXED

    @pytest.mark.asyncio
    async def test_repository_get_existing(self, db_manager):
        """Test getting existing repository."""
        repo_id = "test-repo"
        canonical_path = "/path/to/repo"

        # Create repository
        repo1 = await db_manager.get_or_create_repository(repo_id, canonical_path)

        # Get same repository
        repo2 = await db_manager.get_or_create_repository(repo_id, canonical_path)

        assert repo1.repository_id == repo2.repository_id
        assert repo1.canonical_path == repo2.canonical_path

    @pytest.mark.asyncio
    async def test_repository_status_update(self, db_manager):
        """Test repository status updates."""
        repo_id = "test-repo"
        canonical_path = "/path/to/repo"

        # Create repository
        await db_manager.get_or_create_repository(repo_id, canonical_path)

        # Update status
        await db_manager.update_repository_status(
            repo_id, RepositoryStatus.INDEXING, commit_count=100, progress=50
        )

        # Verify update
        repository = await db_manager.get_or_create_repository(repo_id, canonical_path)
        assert repository.status == RepositoryStatus.INDEXING
        assert repository.commit_count == 100
        assert repository.indexing_progress == 50

    @pytest.mark.asyncio
    async def test_commit_storage(self, db_manager):
        """Test storing commits."""
        repo_id = "test-repo"
        canonical_path = "/path/to/repo"

        # Create repository
        await db_manager.get_or_create_repository(repo_id, canonical_path)

        # Create test commit
        commit = StoredCommit(
            repository_id=repo_id,
            sha="abcd1234",
            embedding=[0.1, 0.2, 0.3, 0.4],
            commit_date=datetime.now(),
            created_at=datetime.now(),
        )

        authors = ["John Doe", "Jane Smith"]

        # Store commit
        await db_manager.store_commit(commit, authors)

        # Verify storage
        exists = await db_manager.commit_exists(repo_id, "abcd1234")
        assert exists is True

        count = await db_manager.get_commit_count(repo_id)
        assert count == 1

    @pytest.mark.asyncio
    async def test_commit_search(self, db_manager):
        """Test commit search functionality."""
        repo_id = "test-repo"
        canonical_path = "/path/to/repo"

        # Create repository
        await db_manager.get_or_create_repository(repo_id, canonical_path)

        # Store test commits
        commits = [
            StoredCommit(
                repository_id=repo_id,
                sha="commit1",
                embedding=[1.0, 0.0, 0.0, 0.0],
                commit_date=datetime.now(),
                created_at=datetime.now(),
            ),
            StoredCommit(
                repository_id=repo_id,
                sha="commit2",
                embedding=[0.0, 1.0, 0.0, 0.0],
                commit_date=datetime.now(),
                created_at=datetime.now(),
            ),
            StoredCommit(
                repository_id=repo_id,
                sha="commit3",
                embedding=[0.5, 0.5, 0.0, 0.0],
                commit_date=datetime.now(),
                created_at=datetime.now(),
            ),
        ]

        for commit in commits:
            await db_manager.store_commit(commit, ["Author"])

        # Search for commits similar to first embedding
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results = await db_manager.search_commits(repo_id, query_embedding, limit=2)

        assert len(results) == 2
        # First result should be exact match
        assert results[0].sha == "commit1"
        assert results[0].similarity_score == 1.0
        # Second should be the mixed embedding
        assert results[1].sha == "commit3"
        assert results[1].similarity_score > 0.5

    @pytest.mark.asyncio
    async def test_cosine_similarity(self, db_manager):
        """Test cosine similarity calculation."""
        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = db_manager._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001

        # Test orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = db_manager._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001

        # Test opposite vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = db_manager._cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.001

    @pytest.mark.asyncio
    async def test_repository_discovered_paths(self, db_manager):
        """Test repository discovered paths functionality."""
        repo_id = "test-repo"
        canonical_path = "/path/to/repo"

        # Create repository
        repository = await db_manager.get_or_create_repository(repo_id, canonical_path)
        assert len(repository.discovered_paths) == 1
        assert canonical_path in repository.discovered_paths

        # Add new discovered path
        new_path = "/another/path/to/same/repo"
        await db_manager.update_repository_discovered_paths(repo_id, new_path)

        # Verify paths updated
        updated_repo = await db_manager.get_or_create_repository(repo_id, canonical_path)
        assert len(updated_repo.discovered_paths) == 2
        assert canonical_path in updated_repo.discovered_paths
        assert new_path in updated_repo.discovered_paths

    @pytest.mark.asyncio
    async def test_database_vacuum(self, db_manager):
        """Test database optimization."""
        # Should run without errors
        await db_manager.vacuum()
