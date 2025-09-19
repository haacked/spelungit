"""
Custom exceptions for the Git History MCP server.
Backward compatibility layer for existing exceptions.
"""

from typing import Optional

# Import from new error system for better functionality
from .errors import DatabaseConnectionError as NewDatabaseConnectionError
from .errors import DatabaseError, EmbeddingError, ErrorCategory, GitOperationError
from .errors import RepositoryIndexingError as NewRepositoryIndexingError
from .errors import RepositoryNotFoundError as NewRepositoryNotFoundError
from .errors import RepositoryNotIndexedError as NewRepositoryNotIndexedError
from .errors import SpelunkError, ValidationError


# Backward compatibility aliases - these maintain the old interface
class GitHistoryMCPException(SpelunkError):
    """Base exception for Git History MCP server."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class RepositoryException(GitHistoryMCPException):
    """Base exception for repository-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.FILESYSTEM, **kwargs)


class RepositoryNotIndexedException(NewRepositoryNotIndexedError):
    """Raised when a repository is not indexed and search is attempted."""

    def __init__(
        self, message: Optional[str] = None, repository_id: Optional[str] = None, **kwargs
    ):
        if repository_id:
            super().__init__(repository_id, **kwargs)
        else:
            # Fallback for old-style usage
            super().__init__("unknown", **kwargs)
            if message:
                self.message = message
                self.user_message = message


class RepositoryIndexingException(NewRepositoryIndexingError):
    """Raised when a repository is currently being indexed."""

    def __init__(
        self, message: Optional[str] = None, repository_id: Optional[str] = None, **kwargs
    ):
        if repository_id:
            super().__init__(repository_id, **kwargs)
        else:
            # Fallback for old-style usage
            super().__init__("unknown", **kwargs)
            if message:
                self.message = message
                self.user_message = message


class RepositoryNotFoundError(NewRepositoryNotFoundError):
    """Raised when a repository cannot be found or accessed."""

    def __init__(
        self, message: Optional[str] = None, repository_id: Optional[str] = None, **kwargs
    ):
        if repository_id:
            super().__init__(repository_id, **kwargs)
        else:
            # Fallback for old-style usage
            super().__init__("unknown", **kwargs)
            if message:
                self.message = message
                self.user_message = message


class InvalidRepositoryError(ValidationError):
    """Raised when a path is not a valid Git repository."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, field="repository_path", **kwargs)


class EmbeddingException(EmbeddingError):
    """Base exception for embedding-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class DatabaseException(DatabaseError):
    """Base exception for database-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class DatabaseConnectionError(NewDatabaseConnectionError):
    """Raised when database connection fails."""

    def __init__(self, message: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if message:
            self.message = message
            self.user_message = message


class DatabaseSchemaError(DatabaseError):
    """Raised when database schema operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, operation="schema_operation", **kwargs)


class SearchException(SpelunkError):
    """Base exception for search-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.INTERNAL, **kwargs)


class SearchQueryError(ValidationError):
    """Raised when search query is invalid."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, field="query", **kwargs)


class GitException(GitOperationError):
    """Base exception for Git operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class GitCommandError(GitOperationError):
    """Raised when Git command execution fails."""

    def __init__(self, message: str, command: Optional[str] = None, **kwargs):
        super().__init__(message, command=command, **kwargs)
