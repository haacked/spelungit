"""
Comprehensive error handling framework with specific exception types,
retry logic, and circuit breaker patterns.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels for proper escalation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better handling strategies."""

    VALIDATION = "validation"
    NETWORK = "network"
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"


@dataclass
class ErrorContext:
    """Context information for errors."""

    operation: str
    component: str
    user_id: Optional[str] = None
    repository_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class SpelunkError(Exception):
    """Base exception for all spelungit errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        user_message: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.cause = cause
        self.user_message = user_message or self._generate_user_message()
        self.timestamp = time.time()

    def _generate_user_message(self) -> str:
        """Generate a user-friendly error message."""
        if self.category == ErrorCategory.VALIDATION:
            return "Invalid input provided. Please check your request and try again."
        elif self.category == ErrorCategory.NETWORK:
            return "Network error occurred. Please check your connection and try again."
        elif self.category == ErrorCategory.DATABASE:
            return "Database error occurred. Please try again later."
        elif self.category == ErrorCategory.AUTHENTICATION:
            return "Authentication failed. Please check your credentials."
        elif self.category == ErrorCategory.AUTHORIZATION:
            return "You don't have permission to perform this action."
        else:
            return "An unexpected error occurred. Please try again later."

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "user_message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "context": {
                "operation": self.context.operation if self.context else None,
                "component": self.context.component if self.context else None,
                "repository_id": self.context.repository_id if self.context else None,
                "additional_data": self.context.additional_data if self.context else None,
            },
            "cause": str(self.cause) if self.cause else None,
        }


# Specific exception types for different error scenarios


class RepositoryError(SpelunkError):
    """Base class for repository-related errors."""

    def __init__(self, repository_id: str, message: str, **kwargs):
        self.repository_id = repository_id
        context = kwargs.get("context") or ErrorContext(
            operation="repository_operation", component="repository", repository_id=repository_id
        )
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class RepositoryNotIndexedError(RepositoryError):
    """Raised when attempting to search a repository that is not indexed."""

    def __init__(self, repository_id: str, commit_count: Optional[int] = None, **kwargs):
        self.commit_count = commit_count
        message = f"Repository '{repository_id}' is not indexed"
        if commit_count:
            message += f". Estimated {commit_count} commits to process"

        super().__init__(
            repository_id,
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            user_message=f"Repository '{repository_id}' needs to be indexed before searching. Use the 'index_repository' tool to begin indexing.",
            **kwargs,
        )


class RepositoryIndexingError(RepositoryError):
    """Raised when repository indexing fails or is in progress."""

    def __init__(self, repository_id: str, progress: Optional[int] = None, **kwargs):
        self.progress = progress
        message = f"Repository '{repository_id}' indexing issue"
        if progress is not None:
            message += f" (progress: {progress}%)"

        super().__init__(
            repository_id,
            message,
            category=ErrorCategory.INTERNAL,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


# Retry logic and circuit breaker patterns


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class RetryableError(SpelunkError):
    """Base class for errors that can be retried."""

    def __init__(self, message: str, retryable: bool = True, **kwargs):
        self.retryable = retryable
        super().__init__(message, **kwargs)


async def retry_async(
    func: Callable[..., Awaitable[T]],
    config: RetryConfig = RetryConfig(),
    retryable_exceptions: tuple = (RetryableError,),
    *args,
    **kwargs,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        config: Retry configuration
        retryable_exceptions: Tuple of exception types that should trigger retry
        *args, **kwargs: Arguments to pass to the function

    Returns:
        The result of the function call

    Raises:
        The last exception if all retry attempts fail
    """
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_attempts - 1:
                logger.error(
                    f"Function {func.__name__} failed after {config.max_attempts} attempts: {e}"
                )
                break

            # Calculate delay with exponential backoff
            delay = min(config.base_delay * (config.exponential_base**attempt), config.max_delay)

            if config.jitter:
                import secrets

                delay *= 0.5 + secrets.SystemRandom().random() * 0.5  # Add 0-50% jitter

            logger.warning(
                f"Function {func.__name__} failed (attempt {attempt + 1}/{config.max_attempts}), retrying in {delay:.2f}s: {e}"
            )
            await asyncio.sleep(delay)

    if last_exception is not None:
        raise last_exception
    else:
        raise RuntimeError(f"Function {func.__name__} failed with no retry attempts")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external service calls.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig = CircuitBreakerConfig()):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.next_attempt_time = 0

    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Call a function through the circuit breaker.

        Args:
            func: The function to call
            *args, **kwargs: Arguments to pass to the function

        Returns:
            The result of the function call

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is open
            The original exception if the function fails
        """
        if self.state == CircuitBreakerState.OPEN:
            if time.time() < self.next_attempt_time:
                raise SpelunkError(
                    f"Circuit breaker '{self.name}' is open",
                    category=ErrorCategory.EXTERNAL_SERVICE,
                    severity=ErrorSeverity.HIGH,
                    user_message="Service is temporarily unavailable. Please try again later.",
                )
            else:
                self.state = CircuitBreakerState.HALF_OPEN

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful function call."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED

    def _on_failure(self):
        """Handle failed function call."""
        self.failure_count += 1
        self.last_failure_time = int(time.time())

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = int(time.time() + self.config.recovery_timeout)
            logger.warning(
                f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
            )


# Error reporting and monitoring


class ErrorReporter(ABC):
    """Abstract base class for error reporting."""

    @abstractmethod
    async def report_error(self, error: SpelunkError) -> None:
        """Report an error to the monitoring system."""
        pass


class LoggingErrorReporter(ErrorReporter):
    """Error reporter that logs errors."""

    async def report_error(self, error: SpelunkError) -> None:
        """Report error by logging."""
        error_dict = error.to_dict()

        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error: {error.message}", extra={"error_data": error_dict})
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {error.message}", extra={"error_data": error_dict})
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(
                f"Medium severity error: {error.message}", extra={"error_data": error_dict}
            )
        else:
            logger.info(f"Low severity error: {error.message}", extra={"error_data": error_dict})


# Global error reporter instance
_error_reporter: Optional[ErrorReporter] = None


def set_error_reporter(reporter: Optional[ErrorReporter]) -> None:
    """Set the global error reporter."""
    global _error_reporter
    _error_reporter = reporter


async def report_error(error: SpelunkError) -> None:
    """Report an error using the global error reporter."""
    if _error_reporter:
        await _error_reporter.report_error(error)
    else:
        # Fallback to basic logging
        logger.error(f"Error: {error.message}", extra={"error_data": error.to_dict()})
