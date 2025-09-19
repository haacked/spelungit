"""Tests for error handling framework."""

import asyncio

import pytest

from spelungit.errors import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    DatabaseError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    LoggingErrorReporter,
    RepositoryNotIndexedError,
    RetryableError,
    RetryConfig,
    SpelunkError,
    ValidationError,
    report_error,
    retry_async,
    set_error_reporter,
)


class TestSpelunkError:
    """Test SpelunkError base class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = SpelunkError("Test error")

        assert error.message == "Test error"
        assert error.category == ErrorCategory.INTERNAL
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context is None
        assert error.cause is None
        assert error.user_message == "An unexpected error occurred. Please try again later."

    def test_error_with_context(self):
        """Test error with context."""
        context = ErrorContext(
            operation="test_operation", component="test_component", repository_id="test_repo"
        )

        error = SpelunkError(
            "Test error",
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            context=context,
        )

        assert error.context is context
        assert error.category == ErrorCategory.DATABASE
        assert error.severity == ErrorSeverity.HIGH

    def test_error_to_dict(self):
        """Test error serialization to dictionary."""
        context = ErrorContext(operation="test_operation", component="test_component")

        error = SpelunkError("Test error", context=context)
        error_dict = error.to_dict()

        assert error_dict["type"] == "SpelunkError"
        assert error_dict["message"] == "Test error"
        assert error_dict["category"] == "internal"
        assert error_dict["severity"] == "medium"
        assert error_dict["context"]["operation"] == "test_operation"
        assert error_dict["context"]["component"] == "test_component"

    def test_user_friendly_messages(self):
        """Test user-friendly message generation."""
        validation_error = SpelunkError("Test", category=ErrorCategory.VALIDATION)
        assert "Invalid input" in validation_error.user_message

        network_error = SpelunkError("Test", category=ErrorCategory.NETWORK)
        assert "Network error" in network_error.user_message

        auth_error = SpelunkError("Test", category=ErrorCategory.AUTHENTICATION)
        assert "Authentication failed" in auth_error.user_message


class TestSpecificErrors:
    """Test specific error types."""

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid field", field="test_field")

        assert error.field == "test_field"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.LOW

    def test_repository_not_indexed_error(self):
        """Test RepositoryNotIndexedError."""
        error = RepositoryNotIndexedError("test_repo", commit_count=100)

        assert error.repository_id == "test_repo"
        assert error.commit_count == 100
        assert "test_repo" in error.message
        assert "100 commits" in error.message

    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Connection failed", operation="connect")

        assert error.operation == "connect"
        assert error.category == ErrorCategory.DATABASE
        assert error.severity == ErrorSeverity.HIGH


class TestRetryLogic:
    """Test retry logic functionality."""

    @pytest.mark.asyncio
    async def test_successful_retry(self):
        """Test successful function call without retry."""
        call_count = 0

        async def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_async(successful_function)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_eventual_success(self):
        """Test retry with eventual success."""
        call_count = 0

        async def eventually_successful_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Temporary failure")
            return "success"

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        result = await retry_async(eventually_successful_function, config)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test retry exhaustion."""
        call_count = 0

        async def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Always fails")

        config = RetryConfig(max_attempts=3, base_delay=0.01)

        with pytest.raises(RetryableError):
            await retry_async(always_failing_function, config)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        call_count = 0

        async def non_retryable_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        config = RetryConfig(max_attempts=3, base_delay=0.01)

        with pytest.raises(ValueError):
            await retry_async(non_retryable_function, config)

        assert call_count == 1


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_successful_calls(self):
        """Test successful calls keep circuit closed."""
        config = CircuitBreakerConfig(failure_threshold=3)
        circuit_breaker = CircuitBreaker("test", config)

        async def successful_function():
            return "success"

        result = await circuit_breaker.call(successful_function)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        circuit_breaker = CircuitBreaker("test", config)

        async def failing_function():
            raise Exception("Failure")

        # First failure - circuit should remain closed (below threshold)
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_function)
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        # Second failure - should open circuit (reaches threshold)
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_function)
        # State changes during the call above due to reaching failure threshold
        assert circuit_breaker.state == CircuitBreakerState.OPEN  # type: ignore[comparison-overlap]

    @pytest.mark.asyncio
    async def test_circuit_half_open_recovery(self):
        """Test circuit recovery through half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01)
        circuit_breaker = CircuitBreaker("test", config)

        async def failing_function():
            raise Exception("Failure")

        async def successful_function():
            return "success"

        # Fail to open circuit
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_function)
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Successful call should close circuit
        result = await circuit_breaker.call(successful_function)
        assert result == "success"
        # State changes from OPEN to CLOSED during successful call above
        assert circuit_breaker.state == CircuitBreakerState.CLOSED  # type: ignore[comparison-overlap]


class TestErrorReporting:
    """Test error reporting functionality."""

    @pytest.mark.asyncio
    async def test_logging_error_reporter(self, caplog):
        """Test logging error reporter."""
        reporter = LoggingErrorReporter()

        error = SpelunkError("Test error", severity=ErrorSeverity.HIGH)

        await reporter.report_error(error)

        assert "High severity error" in caplog.text
        assert "Test error" in caplog.text

    @pytest.mark.asyncio
    async def test_global_error_reporting(self, caplog):
        """Test global error reporting."""
        set_error_reporter(LoggingErrorReporter())

        error = SpelunkError("Global test error")
        await report_error(error)

        assert "Global test error" in caplog.text

    @pytest.mark.asyncio
    async def test_error_reporting_without_reporter(self, caplog):
        """Test error reporting fallback when no reporter set."""
        set_error_reporter(None)

        error = SpelunkError("Fallback test error")
        await report_error(error)

        assert "Fallback test error" in caplog.text
