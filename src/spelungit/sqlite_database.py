"""
SQLite database implementation for the Git History MCP server.
Zero-config alternative to PostgreSQL with vector similarity support.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional numpy import
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .models import Repository, RepositoryStatus, SearchResult, StoredCommit

logger = logging.getLogger(__name__)


class SQLiteDatabaseManager:
    """SQLite-based database manager for zero-config deployment."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to user's config directory
            config_dir = Path.home() / ".config" / "git-history-mcp"
            config_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(config_dir / "git-history.db")

        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        logger.info(f"Using SQLite database: {db_path}")

    async def initialize(self) -> None:
        """Initialize database connection and create schema."""
        # SQLite doesn't support async natively, but we'll keep the async interface
        # for compatibility with PostgreSQL version

        # Ensure parent directory exists and is writable
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open connection with proper settings
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0,  # 30 second timeout
            isolation_level=None,  # Enable autocommit mode
        )
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

        # Configure datetime handling to avoid deprecation warnings
        def adapt_datetime_iso(val):
            """Adapt datetime to ISO string format."""
            return val.isoformat()

        def convert_datetime(val):
            """Convert ISO string back to datetime."""
            return datetime.fromisoformat(val.decode())

        sqlite3.register_adapter(datetime, adapt_datetime_iso)
        sqlite3.register_converter("datetime", convert_datetime)

        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=268435456")  # 256MB

        await self._create_schema()
        logger.info("SQLite database initialized successfully")

    async def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite database connection closed")

    async def _create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()

        try:
            # Check and migrate existing schema
            await self._migrate_schema()

        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            # Continue with table creation for fresh databases

        try:
            # Create repositories table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS repositories (
                    repository_id TEXT PRIMARY KEY,
                    canonical_path TEXT NOT NULL,
                    discovered_paths TEXT NOT NULL DEFAULT '[]',
                    last_indexed TIMESTAMP,
                    commit_count INTEGER,
                    status TEXT DEFAULT 'not_indexed',
                    indexing_progress INTEGER CHECK (indexing_progress >= 0 AND indexing_progress <= 100),
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create git_commits table with vector support
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS git_commits (
                    repository_id TEXT NOT NULL,
                    sha TEXT NOT NULL,
                    embedding TEXT NOT NULL,  -- JSON-stored vector
                    commit_date TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (repository_id, sha),
                    FOREIGN KEY (repository_id) REFERENCES repositories(repository_id) ON DELETE CASCADE
                )
            """
            )

            # Create commit_authors table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS commit_authors (
                    repository_id TEXT NOT NULL,
                    sha TEXT NOT NULL,
                    authors TEXT NOT NULL,  -- JSON-stored array
                    PRIMARY KEY (repository_id, sha),
                    FOREIGN KEY (repository_id, sha) REFERENCES git_commits(repository_id, sha) ON DELETE CASCADE
                )
            """
            )

            # Create indexes for better performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_repositories_status ON repositories(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_repositories_canonical_path ON repositories(canonical_path)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_commits_date ON git_commits(repository_id, commit_date)"
            )

            # Create FTS5 table for text search on commit content (as backup to vector search)
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS commits_fts USING fts5(
                    repository_id,
                    sha,
                    message,
                    diff_content,
                    authors
                )
            """
            )

            self.conn.commit()
            logger.info("SQLite database schema created successfully")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating database schema: {e}")
            raise

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        if HAS_NUMPY:
            # Use numpy for better performance
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            magnitude1 = np.linalg.norm(v1)
            magnitude2 = np.linalg.norm(v2)

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return float(dot_product / (magnitude1 * magnitude2))
        else:
            # Fallback without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return float(dot_product / (magnitude1 * magnitude2))

    # Repository management methods

    async def get_or_create_repository(self, repository_id: str, canonical_path: str) -> Repository:
        """Get existing repository or create a new one."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()

        # Try to get existing repository
        cursor.execute(
            """
            SELECT repository_id, canonical_path, discovered_paths, last_indexed,
                   commit_count, status, indexing_progress, error_message, created_at
            FROM repositories WHERE repository_id = ?
        """,
            (repository_id,),
        )

        row = cursor.fetchone()

        if row:
            return Repository(
                repository_id=row["repository_id"],
                canonical_path=row["canonical_path"],
                discovered_paths=json.loads(row["discovered_paths"] or "[]"),
                last_indexed=(
                    datetime.fromisoformat(row["last_indexed"]) if row["last_indexed"] else None
                ),
                commit_count=row["commit_count"],
                status=RepositoryStatus(row["status"]),
                indexing_progress=row["indexing_progress"],
                error_message=row["error_message"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
        else:
            # Create new repository
            discovered_paths_json = json.dumps([canonical_path])
            cursor.execute(
                """
                INSERT INTO repositories (repository_id, canonical_path, discovered_paths)
                VALUES (?, ?, ?)
            """,
                (repository_id, canonical_path, discovered_paths_json),
            )

            self.conn.commit()

            return Repository(
                repository_id=repository_id,
                canonical_path=canonical_path,
                discovered_paths=[canonical_path],
                created_at=datetime.now(),
            )

    async def update_repository_discovered_paths(self, repository_id: str, new_path: str) -> None:
        """Add a new discovered path to a repository."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()

        # Get current paths
        cursor.execute(
            "SELECT discovered_paths FROM repositories WHERE repository_id = ?", (repository_id,)
        )
        row = cursor.fetchone()

        if row:
            current_paths = json.loads(row["discovered_paths"] or "[]")
            if new_path not in current_paths:
                current_paths.append(new_path)
                cursor.execute(
                    """
                    UPDATE repositories SET discovered_paths = ? WHERE repository_id = ?
                """,
                    (json.dumps(current_paths), repository_id),
                )
                self.conn.commit()

    async def update_repository_status(
        self,
        repository_id: str,
        status: RepositoryStatus,
        commit_count: Optional[int] = None,
        progress: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update repository status and metadata."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()

        # Build update query dynamically
        updates = ["status = ?"]
        params: List[Any] = [status.value]

        if commit_count is not None:
            updates.append("commit_count = ?")
            params.append(commit_count)

        if progress is not None:
            updates.append("indexing_progress = ?")
            params.append(progress)

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        if status == RepositoryStatus.INDEXED:
            updates.append("last_indexed = CURRENT_TIMESTAMP")

        params.append(repository_id)

        cursor.execute(
            f"""
            UPDATE repositories SET {", ".join(updates)}
            WHERE repository_id = ?
        """,  # nosec B608 - updates contains only safe column names
            params,
        )

        self.conn.commit()

    async def store_commit(
        self,
        commit: StoredCommit,
        authors: List[str],
        message: Optional[str] = None,
        diff_content: Optional[str] = None,
    ) -> None:
        """Store a commit and its authors in the database."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()

        try:
            # Serialize embedding as JSON
            embedding_json = json.dumps(commit.embedding)
            authors_json = json.dumps(authors)

            # Insert commit
            cursor.execute(
                """
                INSERT OR REPLACE INTO git_commits (repository_id, sha, embedding, commit_date, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    commit.repository_id,
                    commit.sha,
                    embedding_json,
                    commit.commit_date,
                    commit.created_at,
                ),
            )

            # Insert authors
            cursor.execute(
                """
                INSERT OR REPLACE INTO commit_authors (repository_id, sha, authors)
                VALUES (?, ?, ?)
            """,
                (commit.repository_id, commit.sha, authors_json),
            )

            # Optionally store in FTS5 for text search optimization
            if message and diff_content:
                await self.store_commit_for_fts(
                    commit.repository_id, commit.sha, message, diff_content, authors
                )

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing commit {commit.sha}: {e}")
            raise

    async def commit_exists(self, repository_id: str, sha: str) -> bool:
        """Check if a commit already exists in the database."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT 1 FROM git_commits WHERE repository_id = ? AND sha = ?
        """,
            (repository_id, sha),
        )
        return cursor.fetchone() is not None

    async def get_commit_count(self, repository_id: Optional[str] = None) -> int:
        """Get the total number of commits in the database."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()

        if repository_id:
            cursor.execute(
                "SELECT COUNT(*) FROM git_commits WHERE repository_id = ?", (repository_id,)
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM git_commits")

        result = cursor.fetchone()
        return int(result[0]) if result else 0

    async def search_commits(
        self,
        repository_id: str,
        query_embedding: List[float],
        limit: int = 10,
        author_filter: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        query_text: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for commits using optimized hybrid vector similarity within a repository."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")

        # Strategy: Use smart filtering to reduce vector search space
        candidate_shas = await self._get_search_candidates(
            repository_id, author_filter, date_range, query_text, limit * 3
        )

        if not candidate_shas:
            return []

        # If we have a small candidate set, do full vector search on candidates only
        if len(candidate_shas) <= 200:
            return await self._vector_search_on_candidates(
                repository_id, query_embedding, candidate_shas, limit
            )

        # For larger sets, fall back to full repository search
        return await self._full_vector_search(
            repository_id, query_embedding, limit, author_filter, date_range
        )

    async def _get_search_candidates(
        self,
        repository_id: str,
        author_filter: Optional[str],
        date_range: Optional[Tuple[datetime, datetime]],
        query_text: Optional[str],
        limit: int,
    ) -> List[str]:
        """Get candidate commit SHAs using filters and text search."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()

        # Build base query with filters
        where_conditions = ["gc.repository_id = ?"]
        params = [repository_id]

        if author_filter:
            where_conditions.append("ca.authors LIKE ?")
            params.append(f"%{author_filter}%")

        if date_range:
            where_conditions.append("gc.commit_date BETWEEN ? AND ?")
            params.append(date_range[0].isoformat())
            params.append(date_range[1].isoformat())

        # If we have query text, try FTS5 first
        if query_text and len(query_text.strip()) > 2:
            fts_candidates = await self._fts_search_candidates(repository_id, query_text, limit)
            if fts_candidates:
                # Filter FTS results by other conditions if needed
                if author_filter or date_range:
                    return await self._filter_candidates_by_metadata(
                        fts_candidates, author_filter, date_range
                    )
                return fts_candidates

        # Fall back to metadata-only filtering, ordered by date (recent first)
        # SAFETY: where_conditions only contains predefined safe column names and operators
        # Never add user input directly to where_conditions array
        where_clause = " AND ".join(where_conditions)
        query = f"""
            SELECT gc.sha
            FROM git_commits gc
            LEFT JOIN commit_authors ca ON gc.repository_id = ca.repository_id AND gc.sha = ca.sha
            WHERE {where_clause}
            ORDER BY gc.commit_date DESC
            LIMIT ?
        """  # nosec B608 - where_clause contains only safe predefined conditions

        params.append(str(limit))
        cursor.execute(query, params)
        return [row["sha"] for row in cursor.fetchall()]

    async def _fts_search_candidates(
        self, repository_id: str, query_text: str, limit: int
    ) -> List[str]:
        """Use FTS5 to find text-matching candidates."""
        if not self.conn:
            return []

        cursor = self.conn.cursor()
        try:
            # Search in FTS5 table if it has data
            cursor.execute(
                """
                SELECT sha FROM commits_fts
                WHERE repository_id = ? AND commits_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """,
                (repository_id, query_text, limit),
            )
            return [row["sha"] for row in cursor.fetchall()]
        except Exception as e:
            logger.debug(f"FTS search failed: {e}")
            return []

    async def _filter_candidates_by_metadata(
        self,
        candidate_shas: List[str],
        author_filter: Optional[str],
        date_range: Optional[Tuple[datetime, datetime]],
    ) -> List[str]:
        """Filter candidate SHAs by metadata conditions."""
        if not candidate_shas:
            return []

        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()
        placeholders = ",".join("?" * len(candidate_shas))

        where_conditions = [f"gc.sha IN ({placeholders})"]
        params = candidate_shas

        if author_filter:
            where_conditions.append("ca.authors LIKE ?")
            params.append(f"%{author_filter}%")

        if date_range:
            where_conditions.append("gc.commit_date BETWEEN ? AND ?")
            params.append(date_range[0].isoformat())
            params.append(date_range[1].isoformat())

        where_clause = " AND ".join(where_conditions)
        query = f"""
            SELECT gc.sha
            FROM git_commits gc
            LEFT JOIN commit_authors ca ON gc.repository_id = ca.repository_id AND gc.sha = ca.sha
            WHERE {where_clause}
        """  # nosec B608 - where_clause contains only safe predefined conditions

        cursor.execute(query, params)
        return [row["sha"] for row in cursor.fetchall()]

    async def _vector_search_on_candidates(
        self,
        repository_id: str,
        query_embedding: List[float],
        candidate_shas: List[str],
        limit: int,
    ) -> List[SearchResult]:
        """Perform vector search on a limited set of candidates (optimized path)."""
        if not candidate_shas:
            return []

        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()
        placeholders = ",".join("?" * len(candidate_shas))

        # SAFETY: placeholders is generated from a fixed pattern, safe for SQL injection
        query = f"""
            SELECT repository_id, sha, embedding
            FROM git_commits
            WHERE repository_id = ? AND sha IN ({placeholders})
        """  # nosec B608 - placeholders is a safe parameterized query

        params = [repository_id] + candidate_shas
        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Calculate similarities for candidates only
        results = []
        if HAS_NUMPY:
            # Optimized numpy batch processing for better performance
            embeddings_matrix = []
            sha_list = []

            for row in rows:
                try:
                    stored_embedding = json.loads(row["embedding"])
                    embeddings_matrix.append(stored_embedding)
                    sha_list.append((row["repository_id"], row["sha"]))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Invalid embedding data for commit {row['sha']}: {e}")
                    continue

            if embeddings_matrix:
                # Vectorized cosine similarity computation
                query_vec = np.array(query_embedding)
                embeddings_array = np.array(embeddings_matrix)

                # Compute similarities for all embeddings at once
                similarities = np.dot(embeddings_array, query_vec) / (
                    np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_vec)
                )

                # Create results
                for (repo_id, sha), similarity in zip(sha_list, similarities):
                    results.append(
                        SearchResult(
                            repository_id=repo_id,
                            sha=sha,
                            similarity_score=float(similarity),
                        )
                    )
        else:
            # Fallback without numpy
            for row in rows:
                try:
                    stored_embedding = json.loads(row["embedding"])
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    results.append(
                        SearchResult(
                            repository_id=row["repository_id"],
                            sha=row["sha"],
                            similarity_score=similarity,
                        )
                    )
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Invalid embedding data for commit {row['sha']}: {e}")
                    continue

        # Sort by similarity and limit
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:limit]

    async def _full_vector_search(
        self,
        repository_id: str,
        query_embedding: List[float],
        limit: int,
        author_filter: Optional[str],
        date_range: Optional[Tuple[datetime, datetime]],
    ) -> List[SearchResult]:
        """Perform full vector search (fallback for large datasets)."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()

        # Build query with optional filters
        where_conditions = ["gc.repository_id = ?"]
        params = [repository_id]

        if author_filter:
            where_conditions.append("ca.authors LIKE ?")
            params.append(f"%{author_filter}%")

        if date_range:
            where_conditions.append("gc.commit_date BETWEEN ? AND ?")
            params.append(date_range[0].isoformat())
            params.append(date_range[1].isoformat())

        where_clause = " AND ".join(where_conditions)

        query = f"""
            SELECT gc.repository_id, gc.sha, gc.embedding
            FROM git_commits gc
            LEFT JOIN commit_authors ca ON gc.repository_id = ca.repository_id AND gc.sha = ca.sha
            WHERE {where_clause}
        """  # nosec B608 - where_clause contains only safe predefined conditions

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Calculate similarities (same as original implementation but with numpy optimization)
        return await self._calculate_similarities_batch(rows, query_embedding, limit)

    async def _calculate_similarities_batch(
        self, rows: List[sqlite3.Row], query_embedding: List[float], limit: int
    ) -> List[SearchResult]:
        """Calculate similarities for a batch of rows with optional numpy optimization."""
        results = []

        if HAS_NUMPY and len(rows) > 10:
            # Use numpy for batch processing when beneficial
            embeddings_matrix = []
            sha_list = []

            for row in rows:
                try:
                    stored_embedding = json.loads(row["embedding"])
                    embeddings_matrix.append(stored_embedding)
                    sha_list.append((row["repository_id"], row["sha"]))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Invalid embedding data for commit {row['sha']}: {e}")
                    continue

            if embeddings_matrix:
                # Vectorized computation
                query_vec = np.array(query_embedding)
                embeddings_array = np.array(embeddings_matrix)

                # Batch cosine similarity
                similarities = np.dot(embeddings_array, query_vec) / (
                    np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_vec)
                )

                # Create results
                for (repo_id, sha), similarity in zip(sha_list, similarities):
                    results.append(
                        SearchResult(
                            repository_id=repo_id,
                            sha=sha,
                            similarity_score=float(similarity),
                        )
                    )
        else:
            # Standard processing for small batches or when numpy unavailable
            for row in rows:
                try:
                    stored_embedding = json.loads(row["embedding"])
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    results.append(
                        SearchResult(
                            repository_id=row["repository_id"],
                            sha=row["sha"],
                            similarity_score=similarity,
                        )
                    )
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Invalid embedding data for commit {row['sha']}: {e}")
                    continue

        # Sort by similarity and limit
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:limit]

    async def get_latest_commit_date(self, repository_id: str) -> Optional[datetime]:
        """Get the date of the latest indexed commit for a repository."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT MAX(commit_date) FROM git_commits WHERE repository_id = ?
        """,
            (repository_id,),
        )

        result = cursor.fetchone()[0]
        return datetime.fromisoformat(result) if result else None

    async def get_latest_commit_sha(self, repository_id: str) -> Optional[str]:
        """Get the SHA of the latest indexed commit for a repository."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT sha FROM git_commits
            WHERE repository_id = ?
            ORDER BY commit_date DESC
            LIMIT 1
        """,
            (repository_id,),
        )

        result = cursor.fetchone()
        return result[0] if result else None

    async def get_commits_needing_indexing(
        self, repository_id: str, limit: int = 1000
    ) -> List[str]:
        """Get commit SHAs that need to be indexed (for incremental processing)."""
        # This will be implemented when we add Git integration
        return []

    # Additional methods for better SQLite performance

    async def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()
        cursor.execute("VACUUM")
        self.conn.commit()
        logger.info("Database optimized with VACUUM")

    async def get_database_info(self) -> dict:
        """Get database statistics and information."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()

        # Get table sizes
        cursor.execute("SELECT COUNT(*) FROM repositories")
        repo_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM git_commits")
        commit_count = cursor.fetchone()[0]

        # Get database file size
        try:
            db_size = Path(self.db_path).stat().st_size
            db_size_mb = db_size / (1024 * 1024)
        except (OSError, FileNotFoundError):
            db_size_mb = 0

        return {
            "database_type": "SQLite",
            "database_path": self.db_path,
            "database_size_mb": round(db_size_mb, 2),
            "repositories": repo_count,
            "total_commits": commit_count,
            "supports_vector_search": True,
            "supports_fts": True,
        }

    async def _migrate_schema(self) -> None:
        """Migrate existing schema to add missing columns."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        cursor = self.conn.cursor()

        # Check if git_commits table exists and has created_at column
        cursor.execute("PRAGMA table_info(git_commits)")
        columns = [row[1] for row in cursor.fetchall()]

        if "git_commits" in [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]:
            if "created_at" not in columns:
                logger.info("Adding created_at column to git_commits table")
                cursor.execute(
                    """
                    ALTER TABLE git_commits
                    ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """
                )
                # Update existing records to have created_at values
                cursor.execute(
                    """
                    UPDATE git_commits
                    SET created_at = CURRENT_TIMESTAMP
                    WHERE created_at IS NULL
                """
                )

        # Check repositories table
        cursor.execute("PRAGMA table_info(repositories)")
        repo_columns = [row[1] for row in cursor.fetchall()]

        if "repositories" in [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]:
            if "created_at" not in repo_columns:
                logger.info("Adding created_at column to repositories table")
                cursor.execute(
                    """
                    ALTER TABLE repositories
                    ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """
                )
                cursor.execute(
                    """
                    UPDATE repositories
                    SET created_at = CURRENT_TIMESTAMP
                    WHERE created_at IS NULL
                """
                )

        # Autocommit mode enabled, no explicit commit needed

    async def store_commit_for_fts(
        self,
        repository_id: str,
        sha: str,
        message: str,
        diff_content: str,
        authors: List[str],
    ) -> None:
        """Store commit data in FTS5 table for text search."""
        if not self.conn:
            return

        cursor = self.conn.cursor()
        try:
            # Truncate diff content if too long
            max_diff_length = 8000
            if len(diff_content) > max_diff_length:
                diff_content = diff_content[:max_diff_length] + "..."

            cursor.execute(
                """
                INSERT OR REPLACE INTO commits_fts (repository_id, sha, message, diff_content, authors)
                VALUES (?, ?, ?, ?, ?)
            """,
                (repository_id, sha, message, diff_content, " ".join(authors)),
            )
        except Exception as e:
            logger.debug(f"FTS storage failed for commit {sha}: {e}")

    async def get_authors_for_query(
        self,
        repository_id: str,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get authors who wrote code matching the query, aggregated by relevance.

        Args:
            repository_id: Repository ID to search within
            query_embedding: Embedding vector for semantic search
            query_text: Optional text for FTS search
            limit: Maximum number of authors to return

        Returns:
            List of author information with commit counts and examples
        """
        if not self.conn:
            raise RuntimeError("Database connection not initialized")

        # First get matching commits using existing search logic
        search_results = await self.search_commits(
            repository_id=repository_id,
            query_embedding=query_embedding,
            limit=100,  # Get more commits to aggregate authors from
            query_text=query_text,
        )

        if not search_results:
            return []

        # Get commit SHAs for author lookup
        commit_shas = [result.sha for result in search_results]
        sha_placeholders = ",".join("?" * len(commit_shas))

        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT ca.authors, gc.sha, gc.commit_date
            FROM commit_authors ca
            JOIN git_commits gc ON ca.repository_id = gc.repository_id AND ca.sha = gc.sha
            WHERE ca.repository_id = ? AND ca.sha IN ({sha_placeholders})
            ORDER BY gc.commit_date DESC
        """,  # nosec B608 - sha_placeholders is safely constructed from commit count
            [repository_id] + commit_shas,
        )

        # Aggregate authors with their contributions
        author_stats = {}
        commit_relevance = {result.sha: result.similarity_score for result in search_results}

        for row in cursor.fetchall():
            try:
                authors_list = json.loads(row["authors"])
                commit_sha = row["sha"]
                commit_date = row["commit_date"]
                relevance = commit_relevance.get(commit_sha, 0.0)

                for author in authors_list:
                    if author not in author_stats:
                        author_stats[author] = {
                            "author": author,
                            "commit_count": 0,
                            "total_relevance": 0.0,
                            "max_relevance": 0.0,
                            "commits": [],
                            "latest_commit_date": commit_date,
                        }

                    stats = author_stats[author]
                    stats["commit_count"] += 1
                    stats["total_relevance"] += relevance
                    stats["max_relevance"] = max(stats["max_relevance"], relevance)
                    stats["commits"].append(
                        {"sha": commit_sha, "relevance": relevance, "date": commit_date}
                    )

                    # Keep track of latest commit date
                    if commit_date > stats["latest_commit_date"]:
                        stats["latest_commit_date"] = commit_date

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error processing authors for commit {row['sha']}: {e}")
                continue

        # Sort authors by relevance and contribution
        sorted_authors = sorted(
            author_stats.values(),
            key=lambda x: (x["max_relevance"], x["commit_count"], x["total_relevance"]),
            reverse=True,
        )

        # Format results and add example commits
        result_authors = []
        for author_data in sorted_authors[:limit]:
            # Sort commits by relevance to show best examples
            author_data["commits"].sort(key=lambda c: c["relevance"], reverse=True)

            # Get commit messages for top examples
            top_commits = author_data["commits"][:3]  # Show top 3 commits
            commit_examples = []

            for commit in top_commits:
                cursor.execute(
                    "SELECT message FROM commits_fts WHERE repository_id = ? AND sha = ?",
                    (repository_id, commit["sha"]),
                )
                message_row = cursor.fetchone()
                if message_row:
                    commit_examples.append(
                        {
                            "sha": commit["sha"][:8],
                            "message": message_row["message"],
                            "relevance": commit["relevance"],
                            "date": commit["date"],
                        }
                    )

            result_authors.append(
                {
                    "author": author_data["author"],
                    "commit_count": author_data["commit_count"],
                    "max_relevance": author_data["max_relevance"],
                    "avg_relevance": author_data["total_relevance"] / author_data["commit_count"],
                    "example_commits": commit_examples,
                    "latest_contribution": author_data["latest_commit_date"],
                }
            )

        return result_authors
