"""
Lite embedding system using sentence-transformers + code pattern matching.
No OpenAI API key required - works completely offline.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

# Try to import sentence-transformers, fallback to deterministic embeddings for testing
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class CodePatternExtractor:
    """Extracts meaningful patterns from code diffs and commit messages."""

    def __init__(self):
        # Common programming patterns
        self.function_patterns = [
            r"\bdef\s+(\w+)\(",  # Python functions
            r"\bfunction\s+(\w+)\(",  # JavaScript functions
            r"\b(\w+)\s*\([^)]*\)\s*\{",  # C/Java/JS functions
            r"\bclass\s+(\w+)",  # Class definitions
            r"\binterface\s+(\w+)",  # Interface definitions
            r"\benum\s+(\w+)",  # Enum definitions
        ]

        self.variable_patterns = [
            r"\b(const|let|var)\s+(\w+)",  # JS variables
            r"\b(\w+)\s*=\s*",  # General assignment
        ]

        # Code-related keywords with weights
        self.code_keywords = {
            # Architecture & Patterns
            "api": 2.0,
            "endpoint": 2.0,
            "middleware": 2.0,
            "handler": 2.0,
            "service": 1.8,
            "controller": 1.8,
            "model": 1.8,
            "repository": 1.8,
            "factory": 1.5,
            "singleton": 1.5,
            "adapter": 1.5,
            "wrapper": 1.5,
            # Authentication & Security
            "auth": 2.5,
            "authentication": 2.5,
            "authorization": 2.2,
            "login": 2.2,
            "password": 2.0,
            "token": 2.0,
            "session": 2.0,
            "oauth": 2.0,
            "security": 1.8,
            "permission": 1.8,
            "role": 1.8,
            # Database & Storage
            "database": 2.0,
            "db": 2.0,
            "sql": 2.0,
            "query": 2.0,
            "migration": 2.2,
            "index": 1.8,
            "schema": 1.8,
            "table": 1.5,
            "column": 1.5,
            "cache": 1.8,
            "redis": 1.8,
            "postgres": 1.8,
            "mysql": 1.8,
            # Testing & Quality
            "test": 2.0,
            "testing": 2.0,
            "spec": 1.8,
            "mock": 1.8,
            "fixture": 1.5,
            "stub": 1.5,
            "assertion": 1.5,
            # Error Handling & Logging
            "error": 2.0,
            "exception": 2.0,
            "throw": 1.8,
            "catch": 1.8,
            "logging": 1.8,
            "log": 1.5,
            "debug": 1.5,
            "trace": 1.5,
            # Performance & Infrastructure
            "performance": 1.8,
            "optimize": 1.8,
            "speed": 1.5,
            "memory": 1.5,
            "docker": 1.8,
            "kubernetes": 1.8,
            "deploy": 1.8,
            "build": 1.5,
            # Common operations
            "create": 1.5,
            "update": 1.5,
            "delete": 1.5,
            "insert": 1.5,
            "fetch": 1.5,
            "get": 1.3,
            "set": 1.3,
            "add": 1.3,
            "remove": 1.3,
        }

        # File type weights
        self.file_type_weights = {
            ".py": 1.2,
            ".js": 1.2,
            ".ts": 1.2,
            ".jsx": 1.1,
            ".tsx": 1.1,
            ".java": 1.2,
            ".cpp": 1.2,
            ".c": 1.2,
            ".go": 1.2,
            ".rs": 1.2,
            ".sql": 1.5,
            ".md": 0.8,
            ".txt": 0.7,
            ".json": 0.9,
            ".yaml": 0.9,
        }

    def extract_code_features(
        self, text: str, files_changed: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Extract code-specific features from text and file changes."""
        features = {}
        text_lower = text.lower()

        # Extract function/class names
        functions = set()
        for pattern in self.function_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            functions.update(matches)

        # Weight by function name patterns
        for func in functions:
            features[f"func_{func.lower()}"] = 2.0

        # Extract and weight keywords
        for keyword, weight in self.code_keywords.items():
            if keyword in text_lower:
                # Count occurrences but don't go crazy
                count = min(text_lower.count(keyword), 3)
                features[f"kw_{keyword}"] = weight * count

        # File type analysis
        if files_changed:
            for file_path in files_changed:
                ext = Path(file_path).suffix.lower()
                if ext in self.file_type_weights:
                    features[f"filetype_{ext}"] = self.file_type_weights[ext]

                # Directory patterns
                path_parts = Path(file_path).parts
                for part in path_parts:
                    part_lower = part.lower()
                    if part_lower in ["test", "tests", "spec"]:
                        features["dir_test"] = 1.5
                    elif part_lower in ["auth", "authentication"]:
                        features["dir_auth"] = 2.0
                    elif part_lower in ["api", "routes", "controllers"]:
                        features["dir_api"] = 1.8
                    elif part_lower in ["db", "database", "models"]:
                        features["dir_db"] = 1.8

        return features

    def extract_diff_patterns(self, diff: str) -> Set[str]:
        """Extract meaningful patterns from code diffs."""
        patterns: Set[str] = set()

        if not diff:
            return patterns

        # Added/removed lines
        added_lines = [
            line[1:]
            for line in diff.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        ]
        removed_lines = [
            line[1:]
            for line in diff.split("\n")
            if line.startswith("-") and not line.startswith("---")
        ]

        # Look for significant changes
        for lines, prefix in [(added_lines, "add"), (removed_lines, "del")]:
            for line in lines:
                line_clean = line.strip()
                if not line_clean:
                    continue

                # Function definitions
                for pattern in self.function_patterns:
                    if re.search(pattern, line_clean, re.IGNORECASE):
                        patterns.add(f"{prefix}_function")

                # Import/require statements
                if re.search(r"\b(import|require|from|include)\b", line_clean, re.IGNORECASE):
                    patterns.add(f"{prefix}_import")

                # Return statements
                if re.search(r"\breturn\b", line_clean, re.IGNORECASE):
                    patterns.add(f"{prefix}_return")

                # Error handling
                if re.search(
                    r"\b(throw|catch|except|error|exception)\b", line_clean, re.IGNORECASE
                ):
                    patterns.add(f"{prefix}_error_handling")

                # Logging
                if re.search(r"\b(log|logger|console\.|print)\b", line_clean, re.IGNORECASE):
                    patterns.add(f"{prefix}_logging")

        return patterns


class LiteEmbeddingManager:
    """Embedding manager using sentence-transformers + code pattern matching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional["SentenceTransformer"] = None
        self.code_extractor = CodePatternExtractor()
        self.embedding_dim = 384  # Standard dimension for MiniLM
        self.model_path = Path.home() / ".cache" / "git-history-mcp" / "models"

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("sentence-transformers not available, using fallback embeddings")
            return

        try:
            # Ensure cache directory exists
            self.model_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, cache_folder=str(self.model_path))
            logger.info("✓ Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            self.model = None

    def _fallback_embedding(self, text: str) -> List[float]:
        """Fallback deterministic embedding when sentence-transformers unavailable."""
        # Create a deterministic hash-based embedding
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Convert hex to numbers and normalize
        embedding = []
        for i in range(0, min(len(text_hash), self.embedding_dim * 2), 2):
            hex_pair = text_hash[i : i + 2]
            value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(value * 2 - 1)  # Scale to -1 to 1

        # Pad or truncate to desired dimension
        while len(embedding) < self.embedding_dim:
            embedding.append(0.0)
        embedding = embedding[: self.embedding_dim]

        # Add some text-based features for better relevance
        text_lower = text.lower()
        for i, char in enumerate(text_lower[:50]):  # Use first 50 chars
            if i < len(embedding):
                embedding[i] += (ord(char) / 128.0 - 1) * 0.1  # Small influence

        return embedding

    def format_commit_for_embedding(
        self, message: str, diff: str, files_changed: Optional[List[str]] = None
    ) -> str:
        """Format commit message and diff for embedding generation."""
        # Prioritize commit message as it's usually the clearest intent
        parts = [message]

        if diff:
            # Extract meaningful parts of diff, not the whole thing
            diff_lines = diff.split("\n")

            # Get added/removed lines that look meaningful
            meaningful_lines = []
            for line in diff_lines:
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                    clean_line = line[1:].strip()
                    # Skip very short lines, comments, and whitespace
                    if len(clean_line) > 10 and not clean_line.startswith(("#", "//", "/*")):
                        meaningful_lines.append(clean_line)

            # Include up to 5 meaningful diff lines
            if meaningful_lines:
                parts.append("Code changes: " + " | ".join(meaningful_lines[:5]))

        return " ".join(parts)

    async def generate_embedding(
        self, text: str, files_changed: Optional[List[str]] = None
    ) -> List[float]:
        """Generate hybrid embedding combining semantic + code patterns."""

        # Step 1: Get base semantic embedding
        if self.model:
            try:  # type: ignore[unreachable]
                # Use sentence transformer for semantic understanding
                base_embedding = self.model.encode(text, convert_to_numpy=True)
                base_embedding = base_embedding.tolist()
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                base_embedding = self._fallback_embedding(text)
        else:
            base_embedding = self._fallback_embedding(text)

        # Step 2: Extract code features
        code_features = self.code_extractor.extract_code_features(text, files_changed)

        # Step 3: Create hybrid embedding
        # We'll modify the base embedding based on code patterns
        hybrid_embedding = base_embedding.copy()

        # Boost certain dimensions based on code features
        feature_boost = 0.1  # Small but meaningful boost
        for feature, weight in code_features.items():
            # Use feature name to deterministically select dimensions to boost
            feature_hash = hash(feature) % len(hybrid_embedding)
            hybrid_embedding[feature_hash] += feature_boost * weight

            # Also boost a secondary dimension for redundancy
            secondary_dim = (feature_hash + 42) % len(hybrid_embedding)
            hybrid_embedding[secondary_dim] += feature_boost * weight * 0.5

        # Normalize to prevent embeddings from growing too large
        embedding_magnitude = sum(x * x for x in hybrid_embedding) ** 0.5
        if embedding_magnitude > 0:
            hybrid_embedding = [x / embedding_magnitude for x in hybrid_embedding]

        return hybrid_embedding

    @property
    def model_info(self) -> str:
        """Get model information."""
        if self.model:
            return f"sentence-transformers/{self.model_name} + code patterns"  # type: ignore[unreachable]
        else:
            return "fallback-embeddings + code patterns"
