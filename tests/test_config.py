"""Tests for configuration management."""

import os
from unittest.mock import patch

import pytest

from spelungit.config import (
    ApplicationConfig,
    ConfigManager,
    DatabaseConfig,
    EmbeddingConfig,
    GitConfig,
    MonitoringConfig,
    SearchConfig,
    SecurityConfig,
    ServerConfig,
    _parse_bool,
    get_config_manager,
    reset_config,
)
from spelungit.errors import ConfigurationError


class TestConfigurationDataclasses:
    """Test configuration dataclasses."""

    def test_database_config_defaults(self):
        """Test database configuration defaults."""
        config = DatabaseConfig()

        assert config.connection_timeout == 30.0
        assert config.max_connections == 10
        assert config.enable_wal_mode is True
        assert config.db_path is not None  # Should be set in __post_init__

    def test_embedding_config_defaults(self):
        """Test embedding configuration defaults."""
        config = EmbeddingConfig()

        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.embedding_dim == 384
        assert config.enable_hybrid_search is True
        assert config.cache_dir is not None  # Should be set in __post_init__

    def test_application_config_composition(self):
        """Test application configuration composition."""
        config = ApplicationConfig()

        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.git, GitConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.server, ServerConfig)


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_valid_configuration(self):
        """Test that default configuration is valid."""
        config = ApplicationConfig()
        config.validate()  # Should not raise

    @pytest.mark.parametrize(
        "field_path,invalid_value,expected_error",
        [
            ("database.connection_timeout", -1, "connection timeout must be positive"),
            ("embedding.embedding_dim", 0, "Embedding dimension must be positive"),
            ("monitoring.log_level", "INVALID", "Log level must be one of"),
        ],
    )
    def test_invalid_configuration_values(self, field_path, invalid_value, expected_error):
        """Test validation of invalid configuration values."""
        config = ApplicationConfig()

        # Set the field using dot notation
        obj = config
        field_parts = field_path.split(".")
        for part in field_parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, field_parts[-1], invalid_value)

        with pytest.raises(ConfigurationError, match=expected_error):
            config.validate()

    def test_invalid_search_limits(self):
        """Test validation of invalid search limits (special case with multiple fields)."""
        config = ApplicationConfig()
        config.search.default_limit = 100
        config.search.max_limit = 50

        with pytest.raises(ConfigurationError, match="default limit cannot exceed max limit"):
            config.validate()


class TestEnvironmentVariableLoading:
    """Test loading configuration from environment variables."""

    @patch.dict(
        os.environ,
        {
            "SPELUNK_DB_PATH": "/custom/path/db.sqlite",
            "SPELUNK_DB_TIMEOUT": "45.0",
            "SPELUNK_EMBEDDING_MODEL": "custom-model",
            "SPELUNK_LOG_LEVEL": "DEBUG",
            "SPELUNK_DEV_MODE": "true",
        },
    )
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        config = ApplicationConfig.from_environment()

        assert config.database.db_path == "/custom/path/db.sqlite"
        assert config.database.connection_timeout == 45.0
        assert config.embedding.model_name == "custom-model"
        assert config.monitoring.log_level == "DEBUG"
        assert config.server.enable_development_mode is True

    @patch.dict(os.environ, {"SPELUNK_DB_TIMEOUT": "invalid_float"})
    def test_invalid_environment_variable(self):
        """Test handling of invalid environment variable values."""
        with pytest.raises(ValueError):
            ApplicationConfig.from_environment()


class TestBooleanParsing:
    """Test boolean parsing functionality."""

    def test_parse_bool_true_values(self):
        """Test parsing of true boolean values."""
        true_values = ["true", "True", "TRUE", "1", "yes", "on", "enabled"]

        for value in true_values:
            assert _parse_bool(value) is True

    def test_parse_bool_false_values(self):
        """Test parsing of false boolean values."""
        false_values = ["false", "False", "FALSE", "0", "no", "off", "disabled", "random"]

        for value in false_values:
            assert _parse_bool(value) is False


class TestConfigurationSerialization:
    """Test configuration serialization."""

    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = ApplicationConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "database" in config_dict
        assert "embedding" in config_dict
        assert isinstance(config_dict["database"], dict)
        assert "connection_timeout" in config_dict["database"]

    def test_to_safe_dict_masks_sensitive_fields(self):
        """Test that sensitive fields are masked in safe dictionary."""
        config = ApplicationConfig()
        safe_dict = config.to_safe_dict()

        # Database path should be masked
        assert safe_dict["database"]["db_path"] == "***MASKED***"

    def test_get_sensitive_fields(self):
        """Test getting sensitive field names."""
        config = ApplicationConfig()
        sensitive_fields = config.get_sensitive_fields()

        assert "database.db_path" in sensitive_fields


class TestConfigManager:
    """Test configuration manager."""

    def setUp(self):
        """Set up test environment."""
        reset_config()

    def test_config_manager_singleton(self):
        """Test that config manager is a singleton."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()

        assert manager1 is manager2

    def test_config_caching(self):
        """Test that configuration is cached."""
        manager = ConfigManager()

        config1 = manager.get_config()
        config2 = manager.get_config()

        assert config1 is config2

    def test_config_reload(self):
        """Test configuration reload."""
        manager = ConfigManager()

        config1 = manager.get_config()
        config2 = manager.reload_config()

        assert config1 is not config2

    def test_set_config_for_testing(self):
        """Test setting configuration for testing."""
        manager = ConfigManager()
        test_config = ApplicationConfig()
        test_config.server.name = "test-server"

        manager.set_config(test_config)
        retrieved_config = manager.get_config()

        assert retrieved_config.server.name == "test-server"

    def test_config_section_accessors(self):
        """Test configuration section accessor methods."""
        manager = ConfigManager()

        assert isinstance(manager.get_database_config(), DatabaseConfig)
        assert isinstance(manager.get_embedding_config(), EmbeddingConfig)
        assert isinstance(manager.get_git_config(), GitConfig)
        assert isinstance(manager.get_search_config(), SearchConfig)
        assert isinstance(manager.get_security_config(), SecurityConfig)
        assert isinstance(manager.get_monitoring_config(), MonitoringConfig)
        assert isinstance(manager.get_server_config(), ServerConfig)

    def test_invalid_config_set(self):
        """Test setting invalid configuration raises error."""
        manager = ConfigManager()
        invalid_config = ApplicationConfig()
        invalid_config.database.connection_timeout = -1

        with pytest.raises(ConfigurationError):
            manager.set_config(invalid_config)
