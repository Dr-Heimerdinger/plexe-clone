"""
Common utilities and error handling for the Plexe internal modules.
"""

from plexe.internal.common.errors import (
    PlexeError,
    CodeExecutionError,
    ValidationError,
    RegistryError,
    AgentError,
    DatabaseError,
    DatasetError,
    SchemaError,
    TimeoutError,
    RetryConfig,
    RetryStrategy,
    with_retry,
    ErrorContext,
    log_exception,
    create_error_result,
    format_exception_chain,
    safe_execute,
    DEFAULT_CODE_EXECUTION_RETRY,
    DEFAULT_DATABASE_RETRY,
    DEFAULT_REGISTRY_RETRY,
)

__all__ = [
    "PlexeError",
    "CodeExecutionError",
    "ValidationError",
    "RegistryError",
    "AgentError",
    "DatabaseError",
    "DatasetError",
    "SchemaError",
    "TimeoutError",
    "RetryConfig",
    "RetryStrategy",
    "with_retry",
    "ErrorContext",
    "log_exception",
    "create_error_result",
    "format_exception_chain",
    "safe_execute",
    "DEFAULT_CODE_EXECUTION_RETRY",
    "DEFAULT_DATABASE_RETRY",
    "DEFAULT_REGISTRY_RETRY",
]
