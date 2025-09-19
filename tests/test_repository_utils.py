"""Tests for repository utilities."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from spelungit.repository_utils import (
    estimate_indexing_time,
    generate_repository_id,
    get_canonical_repository_path,
    get_repository_info,
    validate_repository_path,
)


class TestRepositoryUtils:
    """Test repository utility functions."""

    @pytest.fixture
    def temp_git_repo(self):
        """Create a temporary Git repository for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize Git repo
            subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"], cwd=temp_dir, check=True
            )
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_dir, check=True)

            # Create initial commit
            test_file = Path(temp_dir) / "README.md"
            test_file.write_text("Test repository")
            subprocess.run(["git", "add", "README.md"], cwd=temp_dir, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_dir, check=True)

            yield temp_dir

    def test_validate_repository_path_valid(self, temp_git_repo):
        """Test validation of valid Git repository."""
        assert validate_repository_path(temp_git_repo) is True

    def test_validate_repository_path_invalid(self):
        """Test validation of invalid path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Not a Git repo
            assert validate_repository_path(temp_dir) is False

    def test_get_canonical_repository_path(self, temp_git_repo):
        """Test canonical path resolution."""
        canonical_path = get_canonical_repository_path(temp_git_repo)
        assert canonical_path is not None
        assert os.path.exists(canonical_path)

    def test_generate_repository_id_consistency(self, temp_git_repo):
        """Test that repository ID generation is consistent."""
        canonical_path = get_canonical_repository_path(temp_git_repo)
        id1 = generate_repository_id(canonical_path)
        id2 = generate_repository_id(canonical_path)

        assert id1 == id2
        assert isinstance(id1, str)
        assert len(id1) > 0

    def test_get_repository_info(self, temp_git_repo):
        """Test repository information extraction."""
        info = get_repository_info(temp_git_repo)

        assert isinstance(info, dict)
        assert "is_git_repository" in info
        assert "canonical_path" in info
        assert "repository_id" in info
        assert info["is_git_repository"] is True

    def test_estimate_indexing_time(self):
        """Test indexing time estimation."""
        # Test various commit counts
        test_cases = [(50, str), (150, str), (500, str), (1200, str), (5000, str)]

        for commit_count, expected_type in test_cases:
            estimate = estimate_indexing_time(commit_count)
            assert isinstance(estimate, expected_type)
            assert len(estimate) > 0

    def test_repository_info_invalid_path(self):
        """Test repository info for invalid path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            info = get_repository_info(temp_dir)
            assert info["is_git_repository"] is False
