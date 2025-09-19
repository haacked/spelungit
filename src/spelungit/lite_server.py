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
from typing import Any, Dict, List, Optional

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
        git_repo = GitRepository(repository.canonical_path)
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

    async def _estimate_commit_count(self, repository_path: str) -> int:
        """Estimate number of commits in repository."""
        try:
            git_repo = GitRepository(repository_path)
            return await git_repo.get_commit_count()
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
                        created_at=datetime.utcnow(),
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

                if not results:
                    return [TextContent(type="text", text="No matching commits found.")]

                # Format results
                formatted_results = []
                for result in results:
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

                return [
                    TextContent(
                        type="text",
                        text=f"Found {len(results)} matching commits:\n\n"
                        + "\n\n---\n\n".join(formatted_results),
                    )
                ]

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
