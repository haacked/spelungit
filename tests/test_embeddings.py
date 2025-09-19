"""Tests for embedding functionality."""

import pytest

from spelungit.lite_embeddings import CodePatternExtractor, LiteEmbeddingManager


class TestCodePatternExtractor:
    """Test code pattern extraction functionality."""

    @pytest.fixture
    def extractor(self):
        return CodePatternExtractor()

    def test_extract_code_features_basic(self, extractor):
        """Test basic code feature extraction."""
        text = "def authenticate_user(username, password): pass"
        features = extractor.extract_code_features(text)

        assert isinstance(features, dict)
        assert any("func_" in key for key in features.keys())
        assert any("authenticate" in key.lower() for key in features.keys())

    def test_extract_code_features_with_files(self, extractor):
        """Test code features with file information."""
        text = "API authentication changes"
        files = ["src/auth/login.py", "tests/test_auth.py"]

        features = extractor.extract_code_features(text, files)

        assert isinstance(features, dict)
        # Should detect Python files
        assert "filetype_.py" in features
        # Should detect auth directory
        assert "dir_auth" in features
        # Should detect test directory
        assert "dir_test" in features

    def test_extract_diff_patterns(self, extractor):
        """Test diff pattern extraction."""
        diff = """
+def new_function():
+    return True
-def old_function():
-    return False
        """

        patterns = extractor.extract_diff_patterns(diff)

        assert isinstance(patterns, set)
        assert "add_function" in patterns
        assert "del_function" in patterns
        assert "add_return" in patterns
        assert "del_return" in patterns

    def test_extract_diff_patterns_empty(self, extractor):
        """Test diff pattern extraction with empty diff."""
        patterns = extractor.extract_diff_patterns("")
        assert isinstance(patterns, set)
        assert len(patterns) == 0


class TestLiteEmbeddingManager:
    """Test lite embedding manager."""

    @pytest.fixture
    def embedding_manager(self):
        return LiteEmbeddingManager()

    def test_initialization(self, embedding_manager):
        """Test embedding manager initialization."""
        assert embedding_manager.embedding_dim == 384
        assert embedding_manager.code_extractor is not None

    def test_format_commit_for_embedding(self, embedding_manager):
        """Test commit formatting for embedding."""
        message = "Fix authentication bug"
        diff = "+def authenticate():\n+    return True"
        files = ["auth.py"]

        formatted = embedding_manager.format_commit_for_embedding(message, diff, files)

        assert isinstance(formatted, str)
        assert "Fix authentication bug" in formatted
        assert "authenticate" in formatted

    @pytest.mark.asyncio
    async def test_generate_embedding_fallback(self, embedding_manager):
        """Test embedding generation with fallback mode."""
        # Force fallback mode
        embedding_manager.model = None

        text = "Test commit message with authentication"
        embedding = await embedding_manager.generate_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) == embedding_manager.embedding_dim
        assert all(isinstance(x, (int, float)) for x in embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_with_files(self, embedding_manager):
        """Test embedding generation with file context."""
        text = "Add new API endpoint"
        files = ["api/users.py", "tests/test_users.py"]

        embedding = await embedding_manager.generate_embedding(text, files)

        assert isinstance(embedding, list)
        assert len(embedding) == embedding_manager.embedding_dim

    def test_model_info_fallback(self, embedding_manager):
        """Test model info with fallback."""
        # Force fallback mode
        embedding_manager.model = None

        info = embedding_manager.model_info
        assert "fallback-embeddings" in info
        assert "code patterns" in info
