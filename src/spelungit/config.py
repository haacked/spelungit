"""
Configuration management system with validation, environment variable support,
and secure defaults for production deployment.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import ConfigurationError, ErrorSeverity

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    db_path: Optional[str] = None
    connection_timeout: float = 30.0
    max_connections: int = 10
    enable_wal_mode: bool = True
    enable_foreign_keys: bool = True
    vacuum_threshold_mb: int = 100
    backup_enabled: bool = True
    backup_interval_hours: int = 24

    def __post_init__(self):
        if self.db_path is None:
            config_dir = Path.home() / ".config" / "spelungit"
            config_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(config_dir / "git-history.db")


@dataclass
class EmbeddingConfig:
    """Embedding service configuration."""

    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: Optional[str] = None
    embedding_dim: int = 384
    max_text_length: int = 8192
    enable_hybrid_search: bool = True
    code_pattern_weight: float = 0.1
    fallback_enabled: bool = True
    batch_size: int = 32

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".cache" / "spelungit" / "models")


@dataclass
class GitConfig:
    """Git operation configuration."""

    command_timeout: float = 30.0
    diff_size_limit: int = 50000
    commit_batch_size: int = 100
    max_commit_count: int = 100000
    exclude_merge_commits: bool = True
    author_extraction_enabled: bool = True


@dataclass
class SearchConfig:
    """Search configuration settings."""

    default_limit: int = 10
    max_limit: int = 100
    similarity_threshold: float = 0.1
    enable_fts_fallback: bool = True

    # Caching configuration
    cache_enabled: bool = True
    cache_max_entries: int = 1000
    cache_max_memory_mb: int = 100
    cache_max_age_seconds: int = 3600  # 1 hour
    cache_max_idle_seconds: int = 1800  # 30 minutes
    cache_warming_enabled: bool = True

    # Performance optimization
    candidate_filter_threshold: int = 200
    batch_processing_threshold: int = 50
    enable_query_optimization: bool = True


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_concurrent_requests: int = 10
    enable_input_validation: bool = True
    max_query_length: int = 1000
    allowed_file_types: list = field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".sql",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
        ]
    )


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""

    enable_metrics: bool = True
    enable_tracing: bool = False
    log_level: str = "INFO"
    structured_logging: bool = True
    performance_logging: bool = True
    error_reporting: bool = True
    health_check_interval: int = 60


@dataclass
class ServerConfig:
    """MCP server configuration."""

    name: str = "spelungit"
    version: str = "1.0.0"
    max_concurrent_tools: int = 5
    tool_timeout: float = 300.0
    graceful_shutdown_timeout: float = 30.0
    enable_development_mode: bool = False


@dataclass
class ApplicationConfig:
    """Main application configuration."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    git: GitConfig = field(default_factory=GitConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_environment(cls) -> "ApplicationConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Database configuration
        config.database.db_path = os.getenv("SPELUNK_DB_PATH", config.database.db_path)
        config.database.connection_timeout = float(
            os.getenv("SPELUNK_DB_TIMEOUT", config.database.connection_timeout)
        )
        config.database.max_connections = int(
            os.getenv("SPELUNK_DB_MAX_CONNECTIONS", config.database.max_connections)
        )
        config.database.enable_wal_mode = _parse_bool(
            os.getenv("SPELUNK_DB_WAL_MODE", str(config.database.enable_wal_mode))
        )

        # Embedding configuration
        config.embedding.model_name = os.getenv(
            "SPELUNK_EMBEDDING_MODEL", config.embedding.model_name
        )
        config.embedding.cache_dir = os.getenv(
            "SPELUNK_EMBEDDING_CACHE_DIR", config.embedding.cache_dir
        )
        config.embedding.max_text_length = int(
            os.getenv("SPELUNK_MAX_TEXT_LENGTH", config.embedding.max_text_length)
        )
        config.embedding.enable_hybrid_search = _parse_bool(
            os.getenv("SPELUNK_HYBRID_SEARCH", str(config.embedding.enable_hybrid_search))
        )

        # Git configuration
        config.git.command_timeout = float(
            os.getenv("SPELUNK_GIT_TIMEOUT", config.git.command_timeout)
        )
        config.git.diff_size_limit = int(
            os.getenv("SPELUNK_DIFF_SIZE_LIMIT", config.git.diff_size_limit)
        )
        config.git.commit_batch_size = int(
            os.getenv("SPELUNK_COMMIT_BATCH_SIZE", config.git.commit_batch_size)
        )

        # Search configuration
        config.search.default_limit = int(
            os.getenv("SPELUNK_SEARCH_DEFAULT_LIMIT", config.search.default_limit)
        )
        config.search.max_limit = int(
            os.getenv("SPELUNK_SEARCH_MAX_LIMIT", config.search.max_limit)
        )
        config.search.similarity_threshold = float(
            os.getenv("SPELUNK_SIMILARITY_THRESHOLD", config.search.similarity_threshold)
        )
        config.search.cache_enabled = _parse_bool(
            os.getenv("SPELUNK_SEARCH_CACHE", str(config.search.cache_enabled))
        )

        # Security configuration
        config.security.enable_rate_limiting = _parse_bool(
            os.getenv("SPELUNK_RATE_LIMITING", str(config.security.enable_rate_limiting))
        )
        config.security.max_requests_per_minute = int(
            os.getenv("SPELUNK_MAX_REQUESTS_PER_MINUTE", config.security.max_requests_per_minute)
        )
        config.security.max_concurrent_requests = int(
            os.getenv("SPELUNK_MAX_CONCURRENT_REQUESTS", config.security.max_concurrent_requests)
        )

        # Monitoring configuration
        config.monitoring.log_level = os.getenv(
            "SPELUNK_LOG_LEVEL", config.monitoring.log_level
        ).upper()
        config.monitoring.enable_metrics = _parse_bool(
            os.getenv("SPELUNK_ENABLE_METRICS", str(config.monitoring.enable_metrics))
        )
        config.monitoring.enable_tracing = _parse_bool(
            os.getenv("SPELUNK_ENABLE_TRACING", str(config.monitoring.enable_tracing))
        )
        config.monitoring.structured_logging = _parse_bool(
            os.getenv("SPELUNK_STRUCTURED_LOGGING", str(config.monitoring.structured_logging))
        )

        # Server configuration
        config.server.name = os.getenv("SPELUNK_SERVER_NAME", config.server.name)
        config.server.max_concurrent_tools = int(
            os.getenv("SPELUNK_MAX_CONCURRENT_TOOLS", config.server.max_concurrent_tools)
        )
        config.server.tool_timeout = float(
            os.getenv("SPELUNK_TOOL_TIMEOUT", config.server.tool_timeout)
        )
        config.server.enable_development_mode = _parse_bool(
            os.getenv("SPELUNK_DEV_MODE", str(config.server.enable_development_mode))
        )

        # Validate configuration
        config.validate()

        return config

    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []

        # Validate database configuration
        if self.database.connection_timeout <= 0:
            errors.append("Database connection timeout must be positive")

        if self.database.max_connections <= 0:
            errors.append("Database max connections must be positive")

        # Validate embedding configuration
        if self.embedding.embedding_dim <= 0:
            errors.append("Embedding dimension must be positive")

        if self.embedding.max_text_length <= 0:
            errors.append("Max text length must be positive")

        if not (0.0 <= self.embedding.code_pattern_weight <= 1.0):
            errors.append("Code pattern weight must be between 0.0 and 1.0")

        # Validate git configuration
        if self.git.command_timeout <= 0:
            errors.append("Git command timeout must be positive")

        if self.git.diff_size_limit <= 0:
            errors.append("Diff size limit must be positive")

        # Validate search configuration
        if self.search.default_limit <= 0:
            errors.append("Search default limit must be positive")

        if self.search.max_limit <= 0:
            errors.append("Search max limit must be positive")

        if self.search.default_limit > self.search.max_limit:
            errors.append("Search default limit cannot exceed max limit")

        if not (0.0 <= self.search.similarity_threshold <= 1.0):
            errors.append("Similarity threshold must be between 0.0 and 1.0")

        # Validate security configuration
        if self.security.max_requests_per_minute <= 0:
            errors.append("Max requests per minute must be positive")

        if self.security.max_concurrent_requests <= 0:
            errors.append("Max concurrent requests must be positive")

        # Validate monitoring configuration
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.monitoring.log_level not in valid_log_levels:
            errors.append(f"Log level must be one of: {', '.join(valid_log_levels)}")

        # Validate server configuration
        if self.server.max_concurrent_tools <= 0:
            errors.append("Max concurrent tools must be positive")

        if self.server.tool_timeout <= 0:
            errors.append("Tool timeout must be positive")

        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(
                f"- {error}" for error in errors
            )
            raise ConfigurationError(error_message, severity=ErrorSeverity.CRITICAL)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""

        def _dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [_dataclass_to_dict(item) for item in obj]
            else:
                return obj

        return _dataclass_to_dict(self)

    def get_sensitive_fields(self) -> set:
        """Get set of field names that contain sensitive information."""
        return {
            "database.db_path",  # May contain sensitive path information
        }

    def to_safe_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary with sensitive fields masked."""
        config_dict = self.to_dict()
        sensitive_fields = self.get_sensitive_fields()

        def _mask_sensitive(obj, path=""):
            if isinstance(obj, dict):
                return {k: _mask_sensitive(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
            elif path in sensitive_fields:
                return "***MASKED***"
            else:
                return obj

        return _mask_sensitive(config_dict)


def _parse_bool(value: str) -> bool:
    """Parse boolean value from string."""
    return value.lower() in ("true", "1", "yes", "on", "enabled")


class ConfigManager:
    """Configuration manager with caching and validation."""

    def __init__(self):
        self._config: Optional[ApplicationConfig] = None
        self._config_file_path: Optional[str] = None

    def get_config(self) -> ApplicationConfig:
        """Get the current configuration."""
        if self._config is None:
            self._config = ApplicationConfig.from_environment()
            logger.info("Configuration loaded from environment variables")

        return self._config

    def reload_config(self) -> ApplicationConfig:
        """Reload configuration from environment."""
        self._config = None
        return self.get_config()

    def set_config(self, config: ApplicationConfig) -> None:
        """Set configuration (for testing)."""
        config.validate()
        self._config = config

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.get_config().database

    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        return self.get_config().embedding

    def get_git_config(self) -> GitConfig:
        """Get git configuration."""
        return self.get_config().git

    def get_search_config(self) -> SearchConfig:
        """Get search configuration."""
        return self.get_config().search

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return self.get_config().security

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self.get_config().monitoring

    def get_server_config(self) -> ServerConfig:
        """Get server configuration."""
        return self.get_config().server


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> ApplicationConfig:
    """Get the current application configuration."""
    return get_config_manager().get_config()


def reset_config() -> None:
    """Reset configuration (for testing)."""
    global _config_manager
    _config_manager = None
