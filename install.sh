#!/bin/bash
set -e

# Git History MCP Server - Zero Config Installation Script
# This script sets up the MCP server with SQLite + local embeddings

echo "🚀 Installing Git History MCP Server (Zero Config Mode)"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "src/spelungit/__init__.py" ]; then
    print_error "Please run this script from the spelungit repository root directory"
    exit 1
fi

# Check Python version
print_info "Checking Python version..."
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    print_error "Python 3.8+ is required but not found"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

print_status "Python $PYTHON_VERSION found"

# Create virtual environment if it doesn't exist
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv $VENV_DIR
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
print_info "Installing dependencies (this may take a few minutes for first-time setup)..."
pip install -r requirements-lite.txt > /dev/null 2>&1
print_status "Dependencies installed"

# Install the package in development mode
print_info "Installing spelungit package..."
pip install -e . > /dev/null 2>&1
print_status "Package installed in development mode"

# Create configuration directory
CONFIG_DIR="$HOME/.config/spelungit"
mkdir -p "$CONFIG_DIR"
print_status "Configuration directory created at $CONFIG_DIR"

# Find Claude Code configuration
CLAUDE_CONFIG=""
if [ "$(uname)" = "Darwin" ]; then
    # macOS
    CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
elif [ "$(uname)" = "Linux" ]; then
    # Linux
    CLAUDE_CONFIG="$HOME/.config/claude/claude_desktop_config.json"
else
    print_warning "Unsupported operating system. You'll need to manually configure Claude Code."
fi

# Check if Claude Code config exists
if [ -n "$CLAUDE_CONFIG" ] && [ -f "$CLAUDE_CONFIG" ]; then
    print_info "Found Claude Code configuration at $CLAUDE_CONFIG"

    # Get current directory and Python executable paths
    CURRENT_DIR=$(pwd)
    PYTHON_PATH=$(which python)

    # Create MCP server configuration
    read -p "Do you want to automatically add the MCP server to Claude Code? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Backup existing config
        cp "$CLAUDE_CONFIG" "$CLAUDE_CONFIG.backup"
        print_status "Backed up existing Claude Code config"

        # Check if config already has mcpServers
        if grep -q '"mcpServers"' "$CLAUDE_CONFIG"; then
            print_info "Adding spelungit to existing mcpServers..."
            # Use Python to safely modify JSON
            $PYTHON_CMD -c "
import json
import os

config_path = '$CLAUDE_CONFIG'
with open(config_path, 'r') as f:
    config = json.load(f)

if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['spelungit'] = {
    'command': '$PYTHON_PATH',
    'args': ['-m', 'spelungit.lite_server'],
    'env': {
        'PYTHONPATH': '$CURRENT_DIR/src'
    }
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print('MCP server added to Claude Code configuration')
"
        else
            print_info "Creating new mcpServers configuration..."
            # Create new mcpServers section
            $PYTHON_CMD -c "
import json
import os

config_path = '$CLAUDE_CONFIG'
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    config = {}

config['mcpServers'] = {
    'spelungit': {
        'command': '$PYTHON_PATH',
        'args': ['-m', 'spelungit.lite_server'],
        'env': {
            'PYTHONPATH': '$CURRENT_DIR/src'
        }
    }
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print('MCP server configuration created')
"
        fi

        print_status "Git History MCP server added to Claude Code!"
        print_info "Restart Claude Code to load the new MCP server"
    else
        print_info "Skipping automatic Claude Code configuration"
        echo
        print_info "To manually configure Claude Code, add this to your claude_desktop_config.json:"
        echo "{"
        echo "  \"mcpServers\": {"
        echo "    \"spelungit\": {"
        echo "      \"command\": \"$PYTHON_PATH\","
        echo "      \"args\": [\"-m\", \"spelungit.lite_server\"],"
        echo "      \"env\": {"
        echo "        \"PYTHONPATH\": \"$CURRENT_DIR/src\""
        echo "      }"
        echo "    }"
        echo "  }"
        echo "}"
    fi
else
    print_warning "Claude Code configuration not found"
    print_info "You'll need to manually add the MCP server to Claude Code"
    echo
    print_info "Add this to your claude_desktop_config.json:"
    echo "{"
    echo "  \"mcpServers\": {"
    echo "    \"spelungit\": {"
    echo "      \"command\": \"$(which python)\","
    echo "      \"args\": [\"-m\", \"spelungit.lite_server\"],"
    echo "      \"env\": {"
    echo "        \"PYTHONPATH\": \"$(pwd)/src\""
    echo "      }"
    echo "    }"
    echo "  }"
    echo "}"
fi

# Create lite server module
print_info "Creating lite server module..."
cat > src/spelungit/lite_server.py << 'EOF'
#!/usr/bin/env python3
"""
Lite MCP server using SQLite + sentence-transformers.
Zero configuration required - just works out of the box.
"""

import asyncio
import logging
import sys
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions

from .sqlite_database import SQLiteDatabaseManager
from .git_scanner import GitScanner
from .lite_embeddings import LiteEmbeddingManager
from .search_engine import SearchEngine
from .models import RepositoryNotIndexedException, RepositoryIndexingException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize components with lite implementations
db_manager = SQLiteDatabaseManager()
embedding_manager = LiteEmbeddingManager()
server = Server("spelungit-lite")

# Will be initialized in main()
search_engine = None

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available MCP tools."""
    return [
        types.Tool(
            name="search_commits",
            description="Search git commit history using natural language queries (works with any repository)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query about code changes",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                    "author_filter": {
                        "type": "string",
                        "description": "Filter results by author email or name",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="search_blame",
            description="Search code blame data using natural language",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query about code authorship",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Optional file path to filter results",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="who_wrote",
            description="Find who wrote code matching a query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about code to find authors",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of authors to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="search_code_changes",
            description="Search code changes/diffs using natural language",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about code changes",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls from the client."""

    if name == "search_commits":
        return await handle_search_commits(**arguments)
    elif name == "search_blame":
        return await handle_search_blame(**arguments)
    elif name == "who_wrote":
        return await handle_who_wrote(**arguments)
    elif name == "search_code_changes":
        return await handle_search_code_changes(**arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def handle_search_commits(query: str, limit: int = 10, author_filter: str = None) -> list[types.TextContent]:
    """Search commits using natural language query."""
    try:
        results = await search_engine.search_commits(
            query=query,
            limit=limit,
            author_filter=author_filter,
            include_details=True
        )

        if not results:
            response_text = f"No commits found matching '{query}'."
        else:
            response_text = search_engine.format_search_results(results, query)

        return [types.TextContent(type="text", text=response_text)]

    except RepositoryNotIndexedException as e:
        return [types.TextContent(
            type="text",
            text=f"Repository not indexed yet. {str(e)} Please wait a moment and try again."
        )]
    except RepositoryIndexingException as e:
        return [types.TextContent(
            type="text",
            text=f"Repository is being indexed. {str(e)}"
        )]
    except Exception as e:
        logger.error(f"Error in search_commits: {e}")
        return [types.TextContent(type="text", text=f"Error searching commits: {str(e)}")]

async def handle_search_blame(query: str, file_path: str = None, limit: int = 10) -> list[types.TextContent]:
    """Search blame data using natural language."""
    try:
        # For now, redirect to who_wrote
        return await handle_who_wrote(query, limit)
    except Exception as e:
        logger.error(f"Error in search_blame: {e}")
        return [types.TextContent(type="text", text=f"Error searching blame data: {str(e)}")]

async def handle_who_wrote(query: str, limit: int = 5) -> list[types.TextContent]:
    """Find authors who wrote code matching the query."""
    try:
        results = await search_engine.search_by_author(query, limit=limit)

        if not results:
            response_text = f"No authors found for query '{query}'."
        else:
            response_lines = [f"Found {len(results)} commits by authors matching '{query}':\n"]

            author_commits = {}
            for result in results:
                if result.commit_info:
                    author = result.commit_info['author_name']
                    if author not in author_commits:
                        author_commits[author] = []
                    author_commits[author].append(result)

            for author, commits in author_commits.items():
                response_lines.append(f"**{author}** - {len(commits)} commits:")
                for commit in commits[:3]:  # Show top 3 commits per author
                    info = commit.commit_info
                    response_lines.append(f"  - {info['sha'][:8]} - {info['subject'][:60]}...")
                if len(commits) > 3:
                    response_lines.append(f"  ... and {len(commits) - 3} more")
                response_lines.append("")

            response_text = "\n".join(response_lines)

        return [types.TextContent(type="text", text=response_text)]

    except (RepositoryNotIndexedException, RepositoryIndexingException) as e:
        return [types.TextContent(
            type="text",
            text=f"Repository indexing required. {str(e)}"
        )]
    except Exception as e:
        logger.error(f"Error in who_wrote: {e}")
        return [types.TextContent(type="text", text=f"Error finding authors: {str(e)}")]

async def handle_search_code_changes(query: str, limit: int = 10) -> list[types.TextContent]:
    """Search code changes using natural language."""
    try:
        results = await search_engine.search_code_changes(query, limit=limit)

        if not results:
            response_text = f"No code changes found matching '{query}'."
        else:
            response_text = search_engine.format_search_results(results, f"code changes: {query}")

        return [types.TextContent(type="text", text=response_text)]

    except (RepositoryNotIndexedException, RepositoryIndexingException) as e:
        return [types.TextContent(
            type="text",
            text=f"Repository indexing required. {str(e)}"
        )]
    except Exception as e:
        logger.error(f"Error in search_code_changes: {e}")
        return [types.TextContent(type="text", text=f"Error searching code changes: {str(e)}")]

async def main():
    """Main entry point for the lite MCP server."""
    global search_engine

    try:
        # Initialize database
        await db_manager.initialize()
        logger.info("Lite Git History MCP Server initialized successfully")

        # Initialize search engine (git_scanner will be created dynamically per request)
        search_engine = SearchEngine(db_manager, None, embedding_manager)

        # Run server
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="spelungit-lite",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options={}, experimental_capabilities={}
                    ),
                ),
            )

    except Exception as e:
        logger.error(f"Failed to run MCP server: {e}")
        sys.exit(1)
    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
EOF

print_status "Lite server module created"

# Test installation
print_info "Testing installation..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, 'src')

try:
    from spelungit.lite_embeddings import LiteEmbeddingManager
    from spelungit.sqlite_database import SQLiteDatabaseManager
    print('✓ All modules imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)

# Test embedding manager initialization
try:
    manager = LiteEmbeddingManager()
    print(f'✓ Embedding manager initialized: {manager.model_info}')
except Exception as e:
    print(f'✗ Embedding manager error: {e}')
"

echo
print_status "Installation completed successfully!"
echo
print_info "🎯 What's next:"
echo "  1. Restart Claude Code to load the MCP server"
echo "  2. Navigate to any Git repository in your terminal"
echo "  3. Start Claude Code and ask: 'What recent changes were made to authentication?'"
echo "  4. The MCP server will automatically detect and index the repository"
echo
print_info "💡 Features available:"
echo "  • Works with any Git repository automatically"
echo "  • No configuration files needed"
echo "  • No external API keys required"
echo "  • Offline-capable with local embeddings"
echo "  • Git worktree support"
echo
print_info "📊 Storage location: $CONFIG_DIR"
print_info "🔧 Virtual environment: $VENV_DIR"
echo
print_status "Ready to explore your Git history with Claude Code! 🚀"
