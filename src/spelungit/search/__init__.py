"""
Search optimization layer for Spelunk Git MCP.

This package contains advanced search optimization components:
- cache: Multi-layered result caching with LRU eviction
- optimized_search: High-performance search engine with intelligent optimization
- analytics: Search performance monitoring and metrics collection
"""

from .analytics import QueryMetrics, RepositoryAnalytics, SearchAnalytics
from .cache import CacheEntry, CacheKey, SearchResultCache
from .optimized_search import OptimizedSearchEngine

__all__ = [
    "SearchResultCache",
    "CacheKey",
    "CacheEntry",
    "OptimizedSearchEngine",
    "SearchAnalytics",
    "QueryMetrics",
    "RepositoryAnalytics",
]
