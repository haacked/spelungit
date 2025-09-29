"""
Repository utilities for handling Git worktrees and canonical paths.
Provides functions to properly identify repositories regardless of worktree location.
"""

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .models import Repository

logger = logging.getLogger(__name__)


def get_canonical_repository_path(working_dir: str) -> str:
    """
    Get the canonical repository path, handling Git worktrees correctly.

    Args:
        working_dir: Any directory within a Git repository or worktree

    Returns:
        The canonical path to the main repository

    Raises:
        ValueError: If the directory is not within a Git repository
    """
    working_dir = os.path.abspath(working_dir)

    try:
        # First check if this is a Git repository at all
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        # Get the .git directory path (handles both regular repos and worktrees)
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        git_dir = result.stdout.strip()

        # Handle different Git directory structures
        if git_dir == ".git":
            # Simple case: we're in the main repository
            return working_dir
        elif git_dir.endswith("/.git"):
            # We're in the main repository, git-dir is absolute path
            return os.path.dirname(git_dir)
        elif "/worktrees/" in git_dir:
            # This is a worktree - need to find the main repository
            logger.debug(f"Detected worktree git directory: {git_dir}")

            # Try to get the main working tree (Git 2.25+)
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--show-superproject-working-tree"],
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                superproject = result.stdout.strip()
                if superproject:
                    logger.debug(f"Found superproject: {superproject}")
                    return superproject
            except subprocess.CalledProcessError:
                logger.debug("No superproject found, using alternative method")

            # Fallback: parse commondir to find main repository
            common_dir_file = os.path.join(git_dir, "commondir")
            if os.path.exists(common_dir_file):
                with open(common_dir_file, "r") as f:
                    common_path = f.read().strip()

                # commondir is relative to the worktree's git directory
                if not os.path.isabs(common_path):
                    common_path = os.path.join(git_dir, common_path)

                # The common path points to the main repo's .git directory
                main_repo_path = os.path.dirname(common_path)
                logger.debug(f"Found main repository via commondir: {main_repo_path}")
                return main_repo_path

            # Last resort: extract from worktree path structure
            # /path/to/repo/.git/worktrees/branch-name -> /path/to/repo
            git_base = git_dir.split("/.git/worktrees/")[0]
            if os.path.exists(os.path.join(git_base, ".git")):
                logger.debug(f"Found main repository via path parsing: {git_base}")
                return git_base

        # Default case: use show-toplevel to get repository root
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        toplevel = result.stdout.strip()
        logger.debug(f"Using show-toplevel result: {toplevel}")
        return toplevel

    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"Not a Git repository or unable to determine repository: {working_dir}"
        ) from e
    except Exception as e:
        raise ValueError(f"Error processing repository path {working_dir}: {e}") from e


def generate_repository_id(canonical_path: str) -> str:
    """
    Generate a stable, human-readable repository ID from canonical path.

    Args:
        canonical_path: The canonical repository path

    Returns:
        Repository ID in format: "parent/repo-name-hash8"
    """
    normalized_path = os.path.realpath(canonical_path)
    path_obj = Path(normalized_path)

    # Get the last two path components for readability
    path_parts = path_obj.parts[-2:] if len(path_obj.parts) >= 2 else path_obj.parts[-1:]
    path_name = "/".join(path_parts)

    # Create a short hash for uniqueness
    path_hash = hashlib.sha256(normalized_path.encode()).hexdigest()[:8]

    return f"{path_name}-{path_hash}"


def get_current_working_directory() -> str:
    """
    Get the current working directory.
    This is a wrapper to make testing easier and handle MCP context in the future.

    Returns:
        Current working directory path
    """
    return os.getcwd()


def get_repository_info(working_dir: str) -> dict:
    """
    Get comprehensive information about a repository.

    Args:
        working_dir: Directory within the repository

    Returns:
        Dictionary with repository information
    """
    try:
        canonical_path = get_canonical_repository_path(working_dir)
        repository_id = generate_repository_id(canonical_path)

        # Get repository name
        repo_name = Path(canonical_path).name

        # Check if current directory is a worktree
        git_dir_result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        is_worktree = "/worktrees/" in git_dir_result.stdout

        return {
            "repository_id": repository_id,
            "canonical_path": canonical_path,
            "current_path": os.path.abspath(working_dir),
            "repository_name": repo_name,
            "is_worktree": is_worktree,
            "valid": True,
            "is_git_repository": True,
        }

    except ValueError as e:
        return {
            "repository_id": None,
            "canonical_path": None,
            "current_path": os.path.abspath(working_dir),
            "repository_name": None,
            "is_worktree": False,
            "valid": False,
            "is_git_repository": False,
            "error": str(e),
        }


async def detect_repository_context(
    db_manager, repository_path: Optional[str] = None
) -> tuple[str, "Repository"]:
    """
    Detect repository context and ensure it's tracked in database.

    This function consolidates the repository detection logic previously duplicated
    in search engine classes.

    Args:
        db_manager: Database manager instance with get_or_create_repository
                   and update_repository_discovered_paths methods
        repository_path: Optional path to repository (defaults to current working directory)

    Returns:
        Tuple of (repository_id, repository)

    Raises:
        ValueError: If repository_path is not a valid Git repository
    """
    if repository_path is None:
        repository_path = get_current_working_directory()

    # Get repository information
    repo_info = get_repository_info(repository_path)
    if not repo_info["valid"]:
        raise ValueError(f"Invalid repository: {repo_info['error']}")

    repository_id = repo_info["repository_id"]
    canonical_path = repo_info["canonical_path"]
    current_path = repo_info["current_path"]

    # Ensure repository is tracked in database
    repository = await db_manager.get_or_create_repository(repository_id, canonical_path)

    # Update discovered paths if needed
    if current_path not in repository.discovered_paths:
        await db_manager.update_repository_discovered_paths(repository_id, current_path)
        repository.discovered_paths.append(current_path)

    return repository_id, repository
