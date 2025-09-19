"""CLI tools for managing the Git History MCP server."""

import argparse
import asyncio
import logging
import sys

from .config import Config
from .database import DatabaseManager
from .embeddings import EmbeddingManager
from .git_operations import GitManager
from .indexer import CommitIndexer

logger = logging.getLogger(__name__)


async def init_database(args):
    """Initialize the database and index repository commits."""
    try:
        # Validate configuration
        config = Config()
        config.validate()

        # Initialize components
        db_manager = DatabaseManager(config.database_url)
        git_manager = GitManager(config.git_repo_path)
        embedding_manager = EmbeddingManager(config.openai_api_key)

        await db_manager.initialize()
        await git_manager.initialize()

        # Get repository stats
        repo_stats = await git_manager.get_repository_stats()
        print(f"Repository: {repo_stats['repository_path']}")
        print(f"Total commits: {repo_stats['total_commits']}")
        print(
            f"Date range: {repo_stats['earliest_commit_date']} to {repo_stats['latest_commit_date']}"
        )

        # Check current database state
        current_count = await db_manager.get_commit_count()
        print(f"Commits already indexed: {current_count}")

        if current_count >= repo_stats["total_commits"]:
            print("✅ Repository is already fully indexed!")
            return

        # Initialize indexer
        indexer = CommitIndexer(db_manager, git_manager, embedding_manager, "default")

        if args.sample_size:
            print(f"\n🚀 Indexing sample of {args.sample_size} commits...")
            await indexer.index_sample_commits(args.sample_size)
        else:
            print(f"\n🚀 Indexing all {repo_stats['total_commits']} commits...")
            await indexer.index_all_commits()

        final_count = await db_manager.get_commit_count()
        print(f"✅ Indexing complete! Total commits indexed: {final_count}")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    finally:
        if "db_manager" in locals():
            await db_manager.close()


async def update_index(args):
    """Update the index with new commits."""
    try:
        config = Config()
        config.validate()

        db_manager = DatabaseManager(config.database_url)
        git_manager = GitManager(config.git_repo_path)
        embedding_manager = EmbeddingManager(config.openai_api_key)

        await db_manager.initialize()
        await git_manager.initialize()

        # Get latest indexed commit date
        latest_date = await db_manager.get_latest_commit_date()

        if latest_date:
            print(f"Finding commits newer than: {latest_date}")
            new_shas = await git_manager.get_commits_since(latest_date)
            print(f"Found {len(new_shas)} new commits to index")
        else:
            print("No previous index found, indexing all commits")
            new_shas = await git_manager.get_all_commit_shas()

        if not new_shas:
            print("✅ Index is up to date!")
            return

        indexer = CommitIndexer(db_manager, git_manager, embedding_manager, "default")
        await indexer.index_commits(new_shas)

        print(f"✅ Indexed {len(new_shas)} new commits")

    except Exception as e:
        logger.error(f"Update failed: {e}")
        sys.exit(1)
    finally:
        if "db_manager" in locals():
            await db_manager.close()


async def stats(args):
    """Show statistics about the indexed repository."""
    try:
        config = Config()

        db_manager = DatabaseManager(config.database_url)
        git_manager = GitManager(config.git_repo_path)

        await db_manager.initialize()
        await git_manager.initialize()

        # Database stats
        db_count = await db_manager.get_commit_count()
        latest_indexed = await db_manager.get_latest_commit_date()

        # Repository stats
        repo_stats = await git_manager.get_repository_stats()

        print("📊 Git History MCP Server Statistics")
        print("=" * 40)
        print(f"Repository Path: {repo_stats['repository_path']}")
        print(f"Total Commits (Git): {repo_stats['total_commits']}")
        print(f"Indexed Commits (DB): {db_count}")
        print(f"Latest Indexed: {latest_indexed or 'None'}")
        print(
            f"Repository Range: {repo_stats['earliest_commit_date']} to {repo_stats['latest_commit_date']}"
        )

        if db_count > 0:
            coverage = (db_count / repo_stats["total_commits"]) * 100
            print(f"Index Coverage: {coverage:.1f}%")

    except Exception as e:
        logger.error(f"Stats failed: {e}")
        sys.exit(1)
    finally:
        if "db_manager" in locals():
            await db_manager.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Git History MCP Server CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database and index commits")
    init_parser.add_argument(
        "--sample-size",
        "-s",
        type=int,
        help="Index only a sample of N recent commits (for testing)",
    )

    # Update command
    subparsers.add_parser("update", help="Update index with new commits")

    # Stats command
    subparsers.add_parser("stats", help="Show repository and index statistics")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    if args.command == "init":
        asyncio.run(init_database(args))
    elif args.command == "update":
        asyncio.run(update_index(args))
    elif args.command == "stats":
        asyncio.run(stats(args))


if __name__ == "__main__":
    main()
