# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Zero-config installation system with SQLite database
- Hybrid embedding system (sentence-transformers + code pattern matching)
- Multi-repository support with automatic Git repository detection
- Git worktree detection and canonical path resolution
- Comprehensive CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality assurance
- Complete test suite with pytest
- Development environment setup with Makefile
- Contributor documentation and guidelines
- Code formatting with Black and Ruff
- Type checking with MyPy
- Security scanning with Bandit

### Changed

- Migrated from PostgreSQL + OpenAI to SQLite + local embeddings for zero-config deployment
- Updated README with zero-config installation instructions
- Enhanced repository detection to handle complex Git setups

### Technical Details

- **Database**: SQLite with JSON-stored vector embeddings
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 + code-aware pattern matching
- **Search**: Cosine similarity with hybrid relevance scoring
- **Architecture**: Async Python with MCP protocol support
- **Testing**: pytest with fixtures for temporary Git repositories
- **CI**: GitHub Actions with multi-Python version testing

## [0.1.0] - Initial Development

### Added

- Initial MCP server implementation
- PostgreSQL database schema with pgvector
- OpenAI embedding integration
- Basic Git repository indexing
- Commit search functionality
- Multi-author support for commits

### Architecture Decisions

- SHA-only storage approach (97% space savings)
- Embedding-based semantic search
- Git integration for on-demand commit details
- Repository isolation for multi-repository support

---

## Contributing

When adding entries to this changelog:

1. **Use the format**: `### Added/Changed/Deprecated/Removed/Fixed/Security`
2. **Be specific**: Describe what changed, not just that something changed
3. **Group related changes**: Keep similar changes together
4. **Link issues**: Reference GitHub issues where applicable
5. **Update unreleased**: Add new changes to the Unreleased section

Example entry:

```markdown
### Added
- New search filter for commit authors ([#123](https://github.com/user/repo/issues/123))
- Support for branch-specific indexing
- Performance improvements for large repositories (processing time reduced by 40%)
```
