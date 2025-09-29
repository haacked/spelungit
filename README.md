# Spelungit MCP Server

A Model Context Protocol (MCP) server for exploring Git commit history using
semantic search. Search through commits with natural language commands like
"Search git history to find out why was this class added?"

## ✨ Features

- **🔍 Semantic Search**: Natural language queries over Git commit history
- **🌳 Multi-Repository Support**: Automatically detects and isolates different repositories
- **⚡ Zero-Config Installation**: Works out of the box with SQLite + local embeddings
- **🎯 Code-Aware Search**: Hybrid embeddings optimized for code changes and technical content
- **📊 Git Worktree Support**: Handles complex Git setups including worktrees
- **🚀 Claude Code Integration**: Automatic configuration for Claude Code

## 🚀 Quick Install

**One-line installation** (recommended for end users):

```bash
curl -sSL https://raw.githubusercontent.com/haacked/spelungit/main/install-remote.sh | bash
```

This command will:

- Download and install Spelungit automatically
- Set up a Python virtual environment
- Install dependencies (sentence-transformers, SQLite, numpy)
- Configure Claude Code automatically
- Test the installation

**Requirements**: Python 3.8+ and curl/wget

## 🔧 Advanced Installation

### Option 1: Clone and Install

For developers or users who prefer to clone the repository:

```bash
git clone https://github.com/haacked/spelungit.git
cd spelungit
./install.sh
```

### Option 2: Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-lite.txt

# Test the installation
python -m spelungit.server --test
```

Add to your Claude Code configuration (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "spelungit": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "spelungit.server"],
      "env": {
        "PYTHONPATH": "/path/to/spelungit/src"
      }
    }
  }
}
```

## 📖 Usage

### Available Tools

1. **`search_commits`** - Search commits using natural language
2. **`index_repository`** - Index a repository for search
3. **`repository_status`** - Check indexing status
4. **`get_database_info`** - View database statistics
5. **`search_blame`** - Search code blame data using natural language
6. **`who_wrote`** - Find authors who wrote code matching a query
7. **`configure_auto_update`** - Configure automatic index update behavior
8. **`get_auto_update_config`** - Get current automatic index update configuration

### Example Queries

```python
# Search for authentication-related changes
search_commits(query="authentication login changes", limit=5)

# Find database migration commits
search_commits(query="database schema migration", author_filter="john")

# Look for bug fixes
search_commits(query="fix error exception handling")

# Search blame information for specific code
search_blame(query="authentication middleware setup")

# Find who wrote specific functionality
who_wrote(query="database connection pooling", limit=3)
```

### First Time Setup

1. **Index your repository**:

   ```python
   index_repository()
   ```

   This processes all commits and creates embeddings for search.

2. **Check status**:

   ```python
   repository_status()
   ```

3. **Start searching**:

   ```python
   search_commits(query="your search query")
   ```

## 🏗️ Architecture

### Zero-Config Design

- **SQLite Database**: No external database setup required
- **Local Embeddings**: Uses sentence-transformers + code pattern matching
- **Automatic Detection**: Discovers repositories and handles Git worktrees
- **Hybrid Search**: Combines semantic understanding with code-specific patterns

### Code-Aware Features

- **Function/Class Detection**: Recognizes code structure changes
- **File Type Weighting**: Prioritizes different file types appropriately
- **Directory Analysis**: Understands project structure (test/, auth/, api/, etc.)
- **Co-author Support**: Handles commits with multiple authors
- **Pattern Extraction**: Identifies meaningful code patterns in diffs

## 📊 Performance

- **Storage Efficient**: Only stores commit SHAs + embeddings (97% space savings vs full content)
- **Fast Search**: Vector similarity search with cosine similarity
- **Incremental Indexing**: Only processes new commits after initial setup
- **Memory Efficient**: Streaming processing for large repositories

## 🔧 Configuration

### Environment Variables

- `GIT_HISTORY_DB_PATH`: Custom database location
- `PYTHONPATH`: Should include the `src` directory

### Database Location

Default: `~/.config/spelungit/git-history.db`

### Auto-Update Configuration

Configure automatic index updates to keep your search index current:

```python
# Enable automatic updates with custom settings
configure_auto_update(
    enable_auto_update=True,
    background_threshold=100,     # Process in background if >100 new commits
    staleness_check_cache_minutes=10  # Check for staleness every 10 minutes
)

# Check current configuration
get_auto_update_config()
```

### Model Information

- **Embedding Model**: Microsoft's all-MiniLM-L6-v2 via sentence-transformers (384 dimensions)
- **Fallback**: Deterministic hash-based embeddings when sentence-transformers unavailable
- **Code Patterns**: 50+ code-specific keywords and patterns

## 🛠️ Development

### Quick Development Setup

```bash
# Clone and set up development environment
git clone https://github.com/haacked/spelungit.git
cd spelungit

# Set up development environment (creates venv, installs deps)
bin/setup --dev

# Run tests
bin/test

# Check code quality
bin/check
```

### Project Structure

```
src/spelungit/
├── lite_server.py          # Main MCP server
├── sqlite_database.py      # SQLite database adapter
├── lite_embeddings.py      # Hybrid embedding system
├── repository_utils.py     # Git repository detection
├── git_integration.py      # Git operations
├── search_engine.py        # Search functionality
├── models.py               # Data models
├── errors.py               # Error definitions
└── __init__.py             # Package initialization

tests/                      # Test suite
requirements-lite.txt       # Zero-config dependencies
install.sh                  # Automatic installer
```

### Running Tests

```bash
# All tests
bin/test

# Test MCP server functionality
bin/dev server
```

### Code Quality

```bash
# Format code
bin/fmt

# All quality checks (format, lint, type check, security)
bin/check
```

### Development Commands

```bash
# Set up development environment
bin/dev setup

# Run tests
bin/dev test

# Test MCP server
bin/dev server
```

## 🐛 Troubleshooting

### Repository Not Indexed

```
Error: Repository 'repo-name' is not indexed. Use the 'index_repository' tool to begin indexing.
```

**Solution**: Run `index_repository()` tool first.

### Installation Issues

**One-line install fails**:

- Check internet connection
- Ensure Python 3.8+ is installed
- Verify curl or wget is available
- Try the advanced installation method below

**Dependencies fail to install**:
You can still use the fallback mode:

```bash
# Test with fallback embeddings
python -m src.spelungit.server --test
```

### Claude Code Configuration

Ensure your Claude Code config includes the correct Python path and PYTHONPATH.

## 🤝 Contributing

We welcome contributions!

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run quality checks: `bin/check`
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Uses [sentence-transformers](https://www.sbert.net) for embeddings
- Implements the [Model Context Protocol](https://modelcontextprotocol.io)
- Optimized for [Claude Code](https://claude.ai/chat) integration
