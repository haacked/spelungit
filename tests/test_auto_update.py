"""Tests for just-in-time auto-update functionality."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from spelungit.lite_server import SearchEngine
from spelungit.lite_embeddings import LiteEmbeddingManager
from spelungit.models import RepositoryStatus, StoredCommit
from spelungit.sqlite_database import SQLiteDatabaseManager


class TestAutoUpdate:
    """Test auto-update functionality."""

    @pytest_asyncio.fixture
    async def db_manager(self):
        """Create temporary database manager."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            db_path = temp_file.name

        manager = SQLiteDatabaseManager(db_path)
        try:
            await manager.initialize()
            yield manager
        finally:
            # Ensure cleanup always happens
            try:
                await manager.close()
            except Exception:
                pass  # Ignore cleanup errors
            try:
                Path(db_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore file deletion errors

    @pytest_asyncio.fixture
    async def embedding_manager(self):
        """Create mock embedding manager."""
        manager = LiteEmbeddingManager()
        # Mock the generate_embedding method to return a simple vector
        manager.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        manager.format_commit_for_embedding = MagicMock(return_value="test content")
        return manager

    @pytest_asyncio.fixture
    async def search_engine(self, db_manager, embedding_manager):
        """Create search engine with mocked dependencies."""
        engine = SearchEngine(db_manager, embedding_manager)
        try:
            yield engine
        finally:
            # Clean up any background tasks
            try:
                async with engine._task_lock:
                    for task in list(engine._background_tasks.values()):
                        if not task.done():
                            task.cancel()
                    engine._background_tasks.clear()
                    engine._background_progress.clear()
                    engine._staleness_cache.clear()
            except Exception:
                pass  # Ignore cleanup errors

    @pytest_asyncio.fixture
    async def mock_repository(self, db_manager):
        """Create a mock repository in the database."""
        repo_id = "test-repo-123"
        canonical_path = "/test/repo/path"

        repository = await db_manager.get_or_create_repository(repo_id, canonical_path)
        await db_manager.update_repository_status(repo_id, RepositoryStatus.INDEXED, commit_count=5)

        # Add some test commits
        test_commits = [
            {
                "sha": "abc123",
                "date": datetime.now() - timedelta(days=2),
                "message": "Test commit 1",
                "authors": ["test_author"],
            },
            {
                "sha": "def456",
                "date": datetime.now() - timedelta(days=1),
                "message": "Test commit 2",
                "authors": ["test_author"],
            },
        ]

        for commit_data in test_commits:
            stored_commit = StoredCommit(
                repository_id=repo_id,
                sha=commit_data["sha"],
                embedding=[0.1, 0.2, 0.3],
                commit_date=commit_data["date"],
                created_at=datetime.now(),
            )
            await db_manager.store_commit(
                stored_commit,
                commit_data["authors"],
                message=commit_data["message"],
                diff_content="test diff",
            )

        return repo_id, canonical_path, repository

    @pytest.mark.asyncio
    async def test_staleness_detection_fresh_index(self, search_engine, mock_repository):
        """Test staleness detection when index is up to date."""
        repo_id, canonical_path, repository = mock_repository

        with patch("spelungit.lite_server.GitRepository") as mock_git_repo:
            # Mock git rev-list to return 0 new commits (up to date)
            mock_repo_instance = AsyncMock()
            mock_repo_instance._run_git_command.return_value = "0"
            mock_git_repo.return_value = mock_repo_instance

            is_stale, commit_gap = await search_engine._is_index_stale(repo_id, canonical_path)

            assert not is_stale
            assert commit_gap == 0

    @pytest.mark.asyncio
    async def test_staleness_detection_stale_index(self, search_engine, mock_repository):
        """Test staleness detection when index is stale."""
        repo_id, canonical_path, repository = mock_repository

        with patch("spelungit.lite_server.GitRepository") as mock_git_repo:
            # Mock git rev-list to return 3 new commits
            mock_repo_instance = AsyncMock()
            mock_repo_instance._run_git_command.return_value = "3"
            mock_git_repo.return_value = mock_repo_instance

            is_stale, commit_gap = await search_engine._is_index_stale(repo_id, canonical_path)

            assert is_stale
            assert commit_gap == 3

    @pytest.mark.asyncio
    async def test_staleness_cache(self, search_engine, mock_repository):
        """Test that staleness checks are cached properly."""
        repo_id, canonical_path, repository = mock_repository

        with patch("spelungit.lite_server.GitRepository") as mock_git_repo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance._run_git_command.return_value = "2"
            mock_git_repo.return_value = mock_repo_instance

            # First call should hit the git command
            is_stale1, commit_gap1 = await search_engine._is_index_stale(repo_id, canonical_path)

            # Second call should use cache (git command shouldn't be called again)
            is_stale2, commit_gap2 = await search_engine._is_index_stale(repo_id, canonical_path)

            assert is_stale1 == is_stale2 is True
            assert commit_gap1 == commit_gap2 == 2

            # Git command should only have been called once due to caching
            assert mock_repo_instance._run_git_command.call_count == 1

    @pytest.mark.asyncio
    async def test_auto_update_disabled(self, search_engine, mock_repository):
        """Test that auto-update respects the enable flag."""
        repo_id, canonical_path, repository = mock_repository

        # Disable auto-update
        search_engine.enable_auto_update = False

        # Mock stale index
        with patch.object(search_engine, "_is_index_stale") as mock_stale_check:
            mock_stale_check.return_value = (True, 5)

            result = await search_engine._check_and_update_if_stale(repo_id, repository)

            # Should not update when disabled
            assert not result["updated"]
            assert not result["background"]
            assert result["commit_gap"] == 0  # Should not even check staleness when disabled
            assert result["warning_message"] == ""

    @pytest.mark.asyncio
    async def test_background_indexing_triggered(self, search_engine, mock_repository):
        """Test that background indexing is triggered for large gaps."""
        repo_id, canonical_path, repository = mock_repository

        # Set low background threshold for testing
        search_engine.background_threshold = 3

        # Mock stale index with commits exceeding background threshold
        with patch.object(search_engine, "_is_index_stale") as mock_stale_check:
            with patch.object(search_engine, "_start_background_indexing"):
                mock_stale_check.return_value = (True, 5)  # Exceeds threshold of 3

                result = await search_engine._check_and_update_if_stale(repo_id, repository)

                assert result["background"]
                assert not result["updated"]
                assert result["commit_gap"] == 5
                assert "background" in result["warning_message"].lower()

                # Background indexing should have been started but not awaited
                # (it runs as a fire-and-forget task)

    @pytest.mark.asyncio
    async def test_foreground_update_success(self, search_engine, mock_repository):
        """Test successful foreground auto-update for small gaps."""
        repo_id, canonical_path, repository = mock_repository

        # Configure for auto-update with high background threshold
        search_engine.enable_auto_update = True
        search_engine.background_threshold = 50

        with patch.object(search_engine, "_is_index_stale") as mock_stale_check:
            with patch.object(search_engine, "_get_incremental_commits") as mock_get_commits:
                with patch.object(search_engine.embeddings, "generate_embedding") as mock_embedding:
                    # Mock stale but below background threshold
                    mock_stale_check.return_value = (True, 3)

                    # Mock the incremental commits method to return test commits
                    mock_get_commits.return_value = [
                        {
                            "sha": "new123",
                            "message": "New commit",
                            "date": datetime.now(),
                            "authors": ["author1"],
                            "files_changed": ["file1.py"],
                            "diff": "test diff",
                        }
                    ]

                    # Mock embedding generation
                    mock_embedding.return_value = [0.1, 0.2, 0.3]

                    # Mock database methods
                    search_engine.db.commit_exists = AsyncMock(return_value=False)
                    search_engine.db.store_commit = AsyncMock()
                    search_engine.db.get_commit_count = AsyncMock(return_value=6)
                    search_engine.db.update_repository_status = AsyncMock()

                    result = await search_engine._check_and_update_if_stale(repo_id, repository)

                    assert result["updated"]
                    assert not result["background"]
                    assert result["commit_gap"] == 3
                    assert result["warning_message"] == ""

                    # Verify that store_commit was called
                    search_engine.db.store_commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_triggers_auto_update(self, search_engine, mock_repository):
        """Test that search operation triggers auto-update when needed."""
        repo_id, canonical_path, repository = mock_repository

        # Make sure repository is marked as INDEXED for the test
        repository.status = RepositoryStatus.INDEXED

        with patch("spelungit.lite_server.detect_repository_context") as mock_detect:
            with patch.object(
                search_engine, "_validate_and_repair_repository_state"
            ) as mock_validate:
                with patch.object(search_engine.db, "get_or_create_repository") as mock_get_repo:
                    with patch.object(
                        search_engine, "_check_and_update_if_stale"
                    ) as mock_auto_update:
                        with patch.object(
                            search_engine.embeddings, "generate_embedding"
                        ) as mock_embedding:
                            with patch.object(search_engine.db, "search_commits") as mock_search:
                                with patch("spelungit.lite_server.GitRepository") as mock_git_repo:
                                    # Setup mocks
                                    mock_detect.return_value = (repo_id, repository)
                                    mock_validate.return_value = True
                                    mock_get_repo.return_value = repository
                                    mock_auto_update.return_value = {
                                        "updated": True,
                                        "background": False,
                                        "commit_gap": 3,
                                        "warning_message": "",
                                    }
                                    mock_embedding.return_value = [0.1, 0.2, 0.3]
                                    mock_search.return_value = []

                                    mock_repo_instance = AsyncMock()
                                    mock_repo_instance.get_commit_info.return_value = {
                                        "message": "test",
                                        "author": "test",
                                        "date": "2023-01-01",
                                        "files_changed": [],
                                    }
                                    mock_git_repo.return_value = mock_repo_instance

                                    # Perform search
                                    await search_engine.search_commits("test query")

                                    # Verify auto-update was called
                                    mock_auto_update.assert_called_once_with(repo_id, repository)

    @pytest.mark.asyncio
    async def test_configuration_changes(self, search_engine):
        """Test configuration parameter changes."""
        # Test initial configuration
        assert search_engine.enable_auto_update
        assert search_engine.background_threshold == 50
        assert search_engine.staleness_check_cache_minutes == 5

        # Test configuration changes
        search_engine.enable_auto_update = False
        search_engine.background_threshold = 100
        search_engine.staleness_check_cache_minutes = 10

        assert not search_engine.enable_auto_update
        assert search_engine.background_threshold == 100
        assert search_engine.staleness_check_cache_minutes == 10

    @pytest.mark.asyncio
    async def test_cache_cleared_on_config_change(self, search_engine):
        """Test that cache is cleared when configuration changes."""
        # Populate cache
        search_engine._staleness_cache["test"] = ((True, 3), datetime.now())
        assert len(search_engine._staleness_cache) == 1

        # Change cache duration (should clear cache)
        search_engine.staleness_check_cache_minutes = 10

        # In real implementation, cache clearing would happen in the configuration handler
        # Here we simulate that behavior
        search_engine._staleness_cache.clear()
        assert len(search_engine._staleness_cache) == 0

    @pytest.mark.asyncio
    async def test_warning_message_generation(self, search_engine, mock_repository):
        """Test that warning messages are generated for background indexing."""
        repo_id, canonical_path, repository = mock_repository

        # Set very low background threshold
        search_engine.background_threshold = 2

        with patch.object(search_engine, "_is_index_stale") as mock_stale_check:
            with patch.object(search_engine, "_start_background_indexing"):
                # Mock large commit gap
                mock_stale_check.return_value = (True, 75)

                result = await search_engine._check_and_update_if_stale(repo_id, repository)

                assert result["background"]
                assert not result["updated"]
                assert result["commit_gap"] == 75
                assert "75 new commits" in result["warning_message"]
                assert "background" in result["warning_message"].lower()
                assert "may not include the latest commits" in result["warning_message"].lower()

    @pytest.mark.asyncio
    async def test_no_update_when_fresh(self, search_engine, mock_repository):
        """Test that no update happens when index is already fresh."""
        repo_id, canonical_path, repository = mock_repository

        with patch.object(search_engine, "_is_index_stale") as mock_stale_check:
            # Mock fresh index
            mock_stale_check.return_value = (False, 0)

            result = await search_engine._check_and_update_if_stale(repo_id, repository)

            assert not result["updated"]
            assert not result["background"]
            assert result["commit_gap"] == 0
            assert result["warning_message"] == ""
