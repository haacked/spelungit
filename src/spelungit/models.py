"""Data models for the Git History MCP server."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

# Try to import pydantic, fall back to dataclass for testing
try:
    from pydantic import BaseModel

    USE_PYDANTIC = True
except ImportError:
    # For testing without pydantic dependency
    BaseModel = object
    USE_PYDANTIC = False


if USE_PYDANTIC:

    class CommitInfo(BaseModel):
        """Information about a Git commit fetched from the repository."""

        sha: str
        message: str
        author_name: str
        author_email: str
        commit_date: datetime
        files_changed: List[str]
        diff: str
        co_authors: List[str]

else:

    @dataclass
    class CommitInfo:
        """Information about a Git commit fetched from the repository."""

        sha: str
        message: str
        author_name: str
        author_email: str
        commit_date: datetime
        files_changed: List[str]
        diff: str
        co_authors: List[str]


class RepositoryStatus(str, Enum):
    """Status of repository indexing."""

    NOT_INDEXED = "not_indexed"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


if USE_PYDANTIC:

    class Repository(BaseModel):
        """Repository information and indexing status."""

        repository_id: str
        canonical_path: str
        discovered_paths: List[str]
        last_indexed: Optional[datetime] = None
        commit_count: Optional[int] = None
        status: RepositoryStatus = RepositoryStatus.NOT_INDEXED
        indexing_progress: Optional[int] = None
        error_message: Optional[str] = None
        created_at: datetime

    class StoredCommit(BaseModel):
        """A commit record stored in the database."""

        repository_id: str
        sha: str
        embedding: List[float]
        commit_date: datetime
        created_at: datetime

    class CommitAuthors(BaseModel):
        """Authors associated with a commit."""

        repository_id: str
        sha: str
        authors: List[str]

    class SearchResult(BaseModel):
        """Result from a semantic search query."""

        repository_id: str
        sha: str
        similarity_score: float
        commit_info: Optional[CommitInfo] = None
        rank: Optional[int] = None  # Will be set during ranking

    class EmbeddingRequest(BaseModel):
        """Request for generating embeddings."""

        sha: str
        text: str
        commit_date: datetime

    class QueryRequest(BaseModel):
        """Request for querying Git history."""

        query: str
        limit: int = 10
        author_filter: Optional[str] = None
        date_range: Optional[tuple[datetime, datetime]] = None

else:

    @dataclass
    class Repository:
        """Repository information and indexing status."""

        repository_id: str
        canonical_path: str
        discovered_paths: List[str] = field(default_factory=list)
        last_indexed: Optional[datetime] = None
        commit_count: Optional[int] = None
        status: RepositoryStatus = RepositoryStatus.NOT_INDEXED
        indexing_progress: Optional[int] = None
        error_message: Optional[str] = None
        created_at: datetime = field(default_factory=datetime.now)

    @dataclass
    class StoredCommit:
        """A commit record stored in the database."""

        repository_id: str
        sha: str
        embedding: List[float]
        commit_date: datetime
        created_at: datetime = field(default_factory=datetime.now)

    @dataclass
    class CommitAuthors:
        """Authors associated with a commit."""

        repository_id: str
        sha: str
        authors: List[str]

    @dataclass
    class SearchResult:
        """Result from a semantic search query."""

        repository_id: str
        sha: str
        similarity_score: float
        commit_info: Optional[CommitInfo] = None
        rank: Optional[int] = None  # Will be set during ranking

    @dataclass
    class EmbeddingRequest:
        """Request for generating embeddings."""

        sha: str
        text: str
        commit_date: datetime

    @dataclass
    class QueryRequest:
        """Request for querying Git history."""

        query: str
        limit: int = 10
        author_filter: Optional[str] = None
        date_range: Optional[tuple] = None
