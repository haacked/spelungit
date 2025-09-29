# Contributing to Git History MCP Server

Thank you for your interest in contributing to the Git History MCP Server! This guide will help you get
started with development and ensure a smooth contribution process.

## 🚀 Quick Start for Contributors

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/git-history-mcp.git
cd git-history-mcp
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Set up development environment (creates venv, installs deps, sets up hooks)
bin/setup --dev
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
bin/test

# Test the application
bin/dev server

# Check code quality
bin/check
```

## 🧪 Running Tests

We have comprehensive test coverage to ensure code quality:

### Run All Tests

```bash
bin/test
```

### Run Specific Test Categories

```bash
# Unit tests only
bin/test --unit

# Integration tests only
bin/test --integration

# Test individual components
python test_repository_utils.py
bin/dev server

# With coverage report (using pytest directly)
pytest --cov=src/git_history_mcp --cov-report=html
```

## 🔍 Code Quality Standards

We maintain high code quality standards with automated checks:

### Formatting

- **Black**: Code formatter (line length: 100)
- **isort**: Import sorting

```bash
# Format code
bin/fmt
```

### Linting

- **Ruff**: Fast Python linter
- **Bandit**: Security vulnerability scanner

```bash
# Check linting
bin/lint

# Fix linting issues automatically
bin/lint --fix
```

### All Quality Checks

```bash
# Run all checks (format, lint, type check, security)
bin/check

# Run all checks with auto-fixing
bin/check --fix
```

### Pre-commit Hooks

We use pre-commit hooks to automatically check code quality:

```bash
# Hooks are installed automatically with bin/setup --dev

# Run hooks manually
pre-commit run --all-files
```

The hooks will run automatically on `git commit` and prevent commits that don't pass quality checks.

## 📝 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following our style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes

```bash
# Run full test suite
bin/test

# Test specific functionality
bin/dev server

# Check code quality
bin/check

# Simulate CI locally
bin/dev ci
```

### 4. Commit Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add new repository indexing feature

- Implement batch processing for large repositories
- Add progress tracking for indexing operations
- Include tests for new functionality"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 🏗️ Project Structure

Understanding the codebase structure:

```
src/git_history_mcp/
├── server.py               # Main MCP server implementation
├── sqlite_database.py      # SQLite database adapter
├── embeddings.py           # Hybrid embedding system
├── repository_utils.py     # Git repository detection utilities
├── git_integration.py      # Git operations wrapper
├── models.py              # Data models and schemas
└── exceptions.py          # Custom exception classes

tests/                     # Test suite
├── test_repository_utils.py
├── test_embeddings.py
├── test_database.py
└── __init__.py

.github/workflows/         # CI/CD configuration
├── ci.yml                # Main CI pipeline

Configuration files:
├── .pre-commit-config.yaml  # Pre-commit hook configuration
├── mypy.ini                # MyPy type checker settings
├── pytest.ini             # Pytest configuration
├── .bandit                # Security scanner settings
├── requirements-lite.txt   # Runtime dependencies
└── requirements-dev.txt    # Development dependencies
```

## 🎯 Areas for Contribution

### High Priority

- **Performance Improvements**: Optimize embedding generation and search
- **Test Coverage**: Add more comprehensive test scenarios
- **Documentation**: Improve code documentation and examples
- **Error Handling**: Better error messages and recovery

### Medium Priority

- **Feature Enhancements**: New search filters and capabilities
- **Integration Tests**: More real-world repository testing
- **Monitoring**: Add logging and metrics
- **Configuration**: More flexible configuration options

### Low Priority

- **UI Improvements**: Better CLI interface
- **Packaging**: Distribution and installation improvements
- **Examples**: More usage examples and tutorials

## 🧪 Testing Guidelines

### Writing Good Tests

1. **Test Behavior, Not Implementation**

   ```python
   # Good - tests the behavior
   def test_search_finds_relevant_commits():
       results = search_commits("authentication bug fix")
       assert len(results) > 0
       assert "auth" in results[0].message.lower()

   # Avoid - tests implementation details
   def test_search_calls_embedding_function():
       with patch('module.generate_embedding') as mock:
           search_commits("query")
           mock.assert_called_once()
   ```

2. **Use Descriptive Test Names**

   ```python
   def test_repository_detection_handles_worktrees()
   def test_embedding_generation_with_code_patterns()
   def test_database_search_returns_sorted_results()
   ```

3. **Test Edge Cases**
   - Empty inputs
   - Large datasets
   - Network failures
   - Invalid configurations

4. **Use Fixtures for Setup**

   ```python
   @pytest.fixture
   async def temp_git_repo():
       # Set up temporary repository
       yield repo_path
       # Clean up
   ```

### Async Testing

Many components use async/await:

```python
@pytest.mark.asyncio
async def test_async_functionality():
    result = await some_async_function()
    assert result is not None
```

## 🔧 Debugging Tips

### Local Development

```bash
# Run with debug logging
PYTHONPATH=src python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from git_history_mcp.server import test_search
import asyncio
asyncio.run(test_search())
"

# Test specific repository
cd /path/to/your/repo
python /path/to/git-history-mcp/src/git_history_mcp/server.py --test
```

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes `src/`
2. **Database Locked**: Close other database connections
3. **Git Errors**: Ensure you're in a valid Git repository
4. **Embedding Issues**: Check if sentence-transformers is installed

## 📊 Performance Considerations

### Optimization Guidelines

1. **Database Queries**: Use batch operations when possible
2. **Embeddings**: Cache embeddings to avoid regeneration
3. **Memory Usage**: Stream large datasets rather than loading all into memory
4. **Git Operations**: Use shallow clones and specific commit ranges

### Profiling

```bash
# Memory profiling
pip install memory-profiler
python -m memory_profiler your_script.py

# Line profiling
pip install line-profiler
kernprof -l -v your_script.py
```

## 🚨 Security Guidelines

1. **No Secrets in Code**: Never commit API keys, passwords, or sensitive data
2. **Validate Inputs**: Always validate user inputs and file paths
3. **Safe Git Operations**: Be careful with shell commands and Git operations
4. **Dependency Management**: Keep dependencies up to date

### Security Scanning

```bash
# Check for vulnerabilities
bandit -r src/
safety check
```

## 📋 Pull Request Checklist

Before submitting your PR, ensure:

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black`, `isort`)
- [ ] Linting passes (`ruff`, `bandit`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes

## 🆘 Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions or discuss ideas
- **Code Review**: Don't hesitate to ask for review feedback

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the
project (MIT).

## 🙏 Recognition

All contributors will be recognized in the project documentation. Thank you for helping make this
project better!

---

Happy contributing! 🎉
