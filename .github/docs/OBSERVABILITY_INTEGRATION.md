# Observability Integration Guide

This document demonstrates how to integrate the comprehensive observability architecture implemented
in Stage 4 of the principal-level refactoring.

## Architecture Overview

The observability system provides four key components:

1. **Distributed Tracing** - Request tracking with correlation IDs
2. **Metrics Collection** - Performance monitoring with Prometheus export
3. **Structured Logging** - JSON logging with context propagation
4. **Health Monitoring** - Component health checks with alerting

## Quick Start Example

```python
import asyncio
from spelungit.observability import (
    get_tracing_context,
    get_metrics_collector,
    get_logger,
    HealthMonitor,
    trace_operation,
    time_histogram,
    with_logging_context
)

async def example_search_operation():
    # Initialize observability components
    tracing = get_tracing_context()
    await tracing.initialize()

    metrics = get_metrics_collector()
    await metrics.initialize()

    logger = get_logger("example", format_json=True)

    # Generate correlation ID for request tracking
    correlation_id = tracing.create_correlation_id()
    tracing.set_correlation_id(correlation_id)

    # Use distributed tracing with automatic metrics and logging
    async with trace_operation("search_commits",
                             correlation_id=correlation_id,
                             tags={"repository": "test-repo"}) as span:

        # Add logging context
        with with_logging_context(repository="test-repo", user_id="user123"):
            logger.info("Starting search operation")

            # Time the operation automatically
            with time_histogram("spelunk_search_duration_seconds",
                              repository_id="test-repo", cache_hit="false"):

                # Simulate search work
                await asyncio.sleep(0.1)

                # Add trace metadata
                span.add_tag("result_count", 42)
                span.add_log("Search completed successfully")

            # Record metrics
            metrics.increment_counter("spelunk_search_requests_total",
                                    repository_id="test-repo", status="success")

            logger.info("Search operation completed", result_count=42)

# Run the example
asyncio.run(example_search_operation())
```

## Integration with Existing Components

### Database Layer Integration

```python
from spelungit.database.vector_search import VectorSearchEngine
from spelungit.observability import trace_operation, get_logger, increment_counter

class ObservableVectorSearchEngine(VectorSearchEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(f"{__name__}.VectorSearchEngine")

    async def search_commits(self, repository_id: str, query_embedding: List[float],
                           limit: int = 10, **kwargs):

        async with trace_operation("vector_search",
                                 tags={"repository_id": repository_id,
                                      "limit": limit}) as span:

            with with_logging_context(repository_id=repository_id):
                self.logger.info("Starting vector search", limit=limit)

                try:
                    # Time the search operation
                    with time_histogram("spelunk_db_operation_duration_seconds",
                                      operation="vector_search"):
                        results = await super().search_commits(
                            repository_id, query_embedding, limit, **kwargs
                        )

                    # Record successful metrics
                    increment_counter("spelunk_db_operations_total",
                                    operation="vector_search", status="success")
                    increment_counter("spelunk_search_requests_total",
                                    repository_id=repository_id, status="success")

                    # Add trace context
                    span.add_tag("result_count", len(results))
                    span.add_tag("cache_used", kwargs.get("enable_cache", True))

                    self.logger.info("Vector search completed",
                                   result_count=len(results))

                    return results

                except Exception as e:
                    # Record error metrics
                    increment_counter("spelunk_db_operations_total",
                                    operation="vector_search", status="error")
                    increment_counter("spelunk_search_requests_total",
                                    repository_id=repository_id, status="error")

                    # Log error with trace context
                    self.logger.error("Vector search failed", exc_info=e)

                    # Mark span as error
                    span.add_tag("error", True)
                    span.add_tag("error.type", type(e).__name__)

                    raise
```

### Cache Integration

```python
from spelungit.search.cache import SearchResultCache
from spelungit.observability import get_metrics_collector, get_logger

class ObservableSearchResultCache(SearchResultCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(f"{__name__}.SearchResultCache")
        self.metrics = get_metrics_collector()

    async def get(self, *args, **kwargs):
        with trace_operation("cache_get") as span:
            result = await super().get(*args, **kwargs)

            # Record cache metrics
            hit = result is not None
            self.metrics.increment_counter("spelunk_cache_operations_total",
                                         operation="get",
                                         result="hit" if hit else "miss")

            span.add_tag("cache_hit", hit)
            span.add_tag("result_count", len(result) if result else 0)

            self.logger.debug("Cache get operation",
                            cache_hit=hit,
                            result_count=len(result) if result else 0)

            return result

    async def put(self, *args, **kwargs):
        with trace_operation("cache_put") as span:
            await super().put(*args, **kwargs)

            # Record cache put metric
            self.metrics.increment_counter("spelunk_cache_operations_total",
                                         operation="put", result="success")

            # Update cache size gauge
            stats = await self.get_stats()
            self.metrics.set_gauge("spelunk_cache_size_bytes",
                                 stats["memory_usage_bytes"],
                                 cache_type="search_results")

            self.logger.debug("Cache put operation completed")
```

### Health Check Setup

```python
from spelungit.observability.health import (
    HealthMonitor, DatabaseHealthCheck, CacheHealthCheck,
    SystemResourcesHealthCheck, create_unhealthy_alert_rule
)

async def setup_health_monitoring(connection_pool, cache, search_engine):
    health_monitor = HealthMonitor(check_interval_seconds=30)

    # Register health checks
    health_monitor.register_health_check(
        DatabaseHealthCheck(connection_pool, timeout_seconds=10)
    )

    health_monitor.register_health_check(
        CacheHealthCheck(cache, timeout_seconds=5)
    )

    health_monitor.register_health_check(
        SystemResourcesHealthCheck(timeout_seconds=5)
    )

    # Add alerting rules
    health_monitor.add_alert_rule(
        create_unhealthy_alert_rule("database",
                                   message="Database health check failed!")
    )

    health_monitor.add_alert_rule(
        create_unhealthy_alert_rule("cache",
                                   message="Cache health check failed!")
    )

    # Add alert handler
    async def log_alert(alert_data):
        logger = get_logger("health_alerts")
        logger.error("Health alert triggered", alert=alert_data)

    health_monitor.add_alert_handler(log_alert)

    await health_monitor.initialize()

    return health_monitor
```

## Server Integration

```python
from spelungit.server import SearchEngine
from spelungit.observability import (
    get_tracing_context, get_metrics_collector, configure_logging,
    LogLevel, trace_operation, increment_counter
)

class ObservableSpelunkServer(SpelunkGitMCPServer):
    async def initialize(self):
        # Configure observability
        configure_logging(level=LogLevel.INFO, format_json=True)

        # Initialize tracing and metrics
        tracing = get_tracing_context()
        await tracing.initialize()

        metrics = get_metrics_collector()
        await metrics.initialize()

        # Initialize server components
        await super().initialize()

        # Set up health monitoring
        self.health_monitor = await setup_health_monitoring(
            self.connection_pool, self.cache, self.search_engine
        )

    async def search_commits(self, repository_id: str, query: str,
                           limit: int = 10, author_filter: str = None):

        # Generate correlation ID for this request
        tracing = get_tracing_context()
        correlation_id = tracing.create_correlation_id()
        tracing.set_correlation_id(correlation_id)

        async with trace_operation("search_commits_request",
                                 correlation_id=correlation_id,
                                 tags={"repository_id": repository_id}) as span:

            with with_logging_context(
                repository_id=repository_id,
                correlation_id=correlation_id,
                query_length=len(query)
            ):

                logger = get_logger(f"{__name__}.SpelunkServer")
                logger.info("Processing search request",
                          query=query[:100], limit=limit)

                # Time the entire request
                with time_histogram("spelunk_request_duration_seconds",
                                  method="search", endpoint="search_commits"):

                    try:
                        results = await super().search_commits(
                            repository_id, query, limit, author_filter
                        )

                        # Record success metrics
                        increment_counter("spelunk_requests_total",
                                        method="search",
                                        endpoint="search_commits",
                                        status="success")

                        # Add span metadata
                        span.add_tag("result_count", len(results))
                        span.add_tag("query_length", len(query))

                        logger.info("Search request completed successfully",
                                  result_count=len(results))

                        return results

                    except Exception as e:
                        # Record error metrics
                        increment_counter("spelunk_requests_total",
                                        method="search",
                                        endpoint="search_commits",
                                        status="error")

                        logger.error("Search request failed", exc_info=e)

                        raise
```

## Monitoring Dashboard Data

### Prometheus Metrics Export

```python
# Get metrics in Prometheus format for monitoring systems
async def export_prometheus_metrics():
    metrics = get_metrics_collector()
    return metrics.get_prometheus_format()

# Example output:
# HELP spelunk_requests_total Total number of requests
# TYPE spelunk_requests_total counter
# spelunk_requests_total{method="search",endpoint="search_commits",status="success"} 142
#
# HELP spelunk_request_duration_seconds Request duration in seconds
# TYPE spelunk_request_duration_seconds histogram
# spelunk_request_duration_seconds_bucket{method="search",endpoint="search_commits",le="0.01"} 12
# spelunk_request_duration_seconds_bucket{method="search",endpoint="search_commits",le="0.1"} 98
# spelunk_request_duration_seconds_count{method="search",endpoint="search_commits"} 142
# spelunk_request_duration_seconds_sum{method="search",endpoint="search_commits"} 8.45
```

### Health Status API

```python
async def get_health_status():
    health_monitor = get_health_monitor()  # Your health monitor instance
    return health_monitor.get_overall_health()

# Example output:
{
    "status": "healthy",
    "message": "All components are healthy",
    "timestamp": 1704067200.123,
    "checks": {
        "database": {
            "status": "healthy",
            "message": "Database is responsive",
            "duration_ms": 12.3,
            "details": {
                "active_connections": 5,
                "total_connections": 10
            }
        },
        "cache": {
            "status": "healthy",
            "message": "Cache is healthy",
            "duration_ms": 2.1,
            "details": {
                "health_score": 95.0,
                "hit_rate": 0.78,
                "memory_usage_mb": 156.7
            }
        }
    },
    "summary": {
        "total_checks": 3,
        "healthy": 3,
        "degraded": 0,
        "unhealthy": 0
    }
}
```

### Trace Export

```python
async def export_traces(trace_ids=None):
    tracing = get_tracing_context()

    # Export in Jaeger format
    jaeger_traces = await tracing.export_traces(format="jaeger", trace_ids=trace_ids)

    # Export in JSON format
    json_traces = await tracing.export_traces(format="json", trace_ids=trace_ids)

    return {
        "jaeger": jaeger_traces,
        "json": json_traces
    }
```

## Performance Impact

The observability system is designed for minimal performance impact:

- **Tracing**: < 1ms overhead per operation
- **Metrics**: Thread-safe counters with minimal locking
- **Logging**: Async-safe with structured JSON output
- **Health Checks**: Configurable intervals, concurrent execution

## Best Practices

1. **Correlation IDs**: Always generate and propagate correlation IDs for request tracking
2. **Trace Context**: Use trace operations for all significant business logic
3. **Metric Labels**: Keep label cardinality reasonable (< 1000 unique combinations)
4. **Log Levels**: Use appropriate log levels (DEBUG for development, INFO for production)
5. **Health Checks**: Configure appropriate timeouts and check intervals
6. **Alert Rules**: Set up meaningful alerts with proper cooldown periods

This comprehensive observability architecture provides principal-level monitoring capabilities
suitable for production deployment and operational excellence.
