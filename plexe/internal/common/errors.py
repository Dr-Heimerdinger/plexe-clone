"""
Centralized error handling, custom exceptions, and retry utilities for the Plexe multi-agent system.

This module provides:
- Custom exception hierarchy for better error categorization
- Retry decorators with configurable backoff for agent operations
- Error context managers for better stack traces
- Utility functions for error formatting and logging
"""

import logging
import traceback
import functools
import time
from typing import Optional, Type, Tuple, Callable, Any, Dict, List, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================

class PlexeError(Exception):
    """
    Base exception for all Plexe errors.
    
    Provides enhanced error context including:
    - Original exception chain
    - Stack trace preservation
    - Error categorization
    - Contextual information
    """
    
    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
    ):
        self.message = message
        self.cause = cause
        self.context = context or {}
        self.recoverable = recoverable
        self.stack_trace = traceback.format_exc()
        
        # Build full message with context
        full_message = self._build_full_message()
        super().__init__(full_message)
    
    def _build_full_message(self) -> str:
        """Build comprehensive error message with all context."""
        parts = [self.message]
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"[Context: {context_str}]")
        
        if self.cause:
            parts.append(f"[Caused by: {type(self.cause).__name__}: {str(self.cause)}]")
        
        return " ".join(parts)
    
    def get_full_trace(self) -> str:
        """Get complete stack trace including cause chain."""
        traces = [f"PlexeError: {self.message}"]
        traces.append(f"Stack trace:\n{self.stack_trace}")
        
        if self.cause:
            traces.append(f"\nCaused by {type(self.cause).__name__}: {str(self.cause)}")
            if hasattr(self.cause, '__traceback__'):
                cause_trace = ''.join(traceback.format_tb(self.cause.__traceback__))
                traces.append(f"Cause stack trace:\n{cause_trace}")
        
        return "\n".join(traces)


class CodeExecutionError(PlexeError):
    """Error during code execution (training, inference, feature transformation)."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        return_code: Optional[int] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        
        # Add execution details to context
        exec_context = context or {}
        if return_code is not None:
            exec_context["return_code"] = return_code
        if stderr:
            # Truncate stderr for context, full version available in attribute
            exec_context["stderr_preview"] = stderr[:200] + "..." if len(stderr) > 200 else stderr
        
        super().__init__(message, cause=cause, context=exec_context, recoverable=True)


class ValidationError(PlexeError):
    """Error during code or data validation."""
    
    def __init__(
        self,
        message: str,
        validation_stage: str,
        validation_details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.validation_stage = validation_stage
        self.validation_details = validation_details or {}
        
        context = {"stage": validation_stage, **self.validation_details}
        super().__init__(message, cause=cause, context=context, recoverable=True)


class RegistryError(PlexeError):
    """Error related to ObjectRegistry operations."""
    
    def __init__(
        self,
        message: str,
        operation: str,
        item_type: Optional[str] = None,
        item_name: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        context = {"operation": operation}
        if item_type:
            context["item_type"] = item_type
        if item_name:
            context["item_name"] = item_name
        
        super().__init__(message, cause=cause, context=context, recoverable=False)


class AgentError(PlexeError):
    """Error during agent execution."""
    
    def __init__(
        self,
        message: str,
        agent_name: str,
        step: Optional[int] = None,
        task: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        context = {"agent": agent_name}
        if step is not None:
            context["step"] = step
        if task:
            context["task"] = task[:100] + "..." if len(task) > 100 else task
        
        super().__init__(message, cause=cause, context=context, recoverable=True)


class DatabaseError(PlexeError):
    """Error related to database operations."""
    
    def __init__(
        self,
        message: str,
        operation: str,
        table: Optional[str] = None,
        query: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        context = {"operation": operation}
        if table:
            context["table"] = table
        if query:
            # Truncate query for safety
            context["query_preview"] = query[:100] + "..." if len(query) > 100 else query
        
        super().__init__(message, cause=cause, context=context, recoverable=True)


class DatasetError(PlexeError):
    """Error related to dataset operations."""
    
    def __init__(
        self,
        message: str,
        dataset_name: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        context = {}
        if dataset_name:
            context["dataset"] = dataset_name
        if operation:
            context["operation"] = operation
        
        super().__init__(message, cause=cause, context=context, recoverable=True)


class SchemaError(PlexeError):
    """Error related to schema operations."""
    
    def __init__(
        self,
        message: str,
        schema_type: Optional[str] = None,
        schema_name: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        context = {}
        if schema_type:
            context["schema_type"] = schema_type
        if schema_name:
            context["schema_name"] = schema_name
        
        super().__init__(message, cause=cause, context=context, recoverable=False)


class TimeoutError(PlexeError):
    """Error when an operation times out."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        operation: str,
        cause: Optional[Exception] = None,
    ):
        context = {"timeout_seconds": timeout_seconds, "operation": operation}
        super().__init__(message, cause=cause, context=context, recoverable=True)


# =============================================================================
# Retry Configuration and Decorators
# =============================================================================

class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential"
    LINEAR_BACKOFF = "linear"
    CONSTANT = "constant"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()
    on_retry_callback: Optional[Callable[[int, Exception, float], None]] = None


# Default configurations for different operation types
DEFAULT_CODE_EXECUTION_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=2.0,
    max_delay=30.0,
    backoff_multiplier=2.0,
    retryable_exceptions=(CodeExecutionError, OSError, IOError),
    non_retryable_exceptions=(ValidationError, SyntaxError),
)

DEFAULT_DATABASE_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=15.0,
    backoff_multiplier=2.0,
    retryable_exceptions=(DatabaseError, ConnectionError, TimeoutError),
)

DEFAULT_REGISTRY_RETRY = RetryConfig(
    max_attempts=2,
    initial_delay=0.5,
    max_delay=5.0,
    retryable_exceptions=(RegistryError, KeyError),
)


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for a retry attempt based on strategy."""
    if config.strategy == RetryStrategy.CONSTANT:
        return config.initial_delay
    elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = config.initial_delay * attempt
    else:  # EXPONENTIAL_BACKOFF
        delay = config.initial_delay * (config.backoff_multiplier ** (attempt - 1))
    
    return min(delay, config.max_delay)


def with_retry(
    config: Optional[RetryConfig] = None,
    max_attempts: Optional[int] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
):
    """
    Decorator to add retry logic to functions.
    
    Args:
        config: Full RetryConfig object for detailed configuration
        max_attempts: Override max attempts (shorthand)
        retryable_exceptions: Override retryable exceptions (shorthand)
    
    Usage:
        @with_retry(max_attempts=3)
        def my_function():
            ...
        
        @with_retry(config=DEFAULT_CODE_EXECUTION_RETRY)
        def execute_code():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build effective config
            effective_config = config or RetryConfig()
            if max_attempts is not None:
                effective_config = RetryConfig(
                    max_attempts=max_attempts,
                    initial_delay=effective_config.initial_delay,
                    max_delay=effective_config.max_delay,
                    backoff_multiplier=effective_config.backoff_multiplier,
                    strategy=effective_config.strategy,
                    retryable_exceptions=retryable_exceptions or effective_config.retryable_exceptions,
                    non_retryable_exceptions=effective_config.non_retryable_exceptions,
                    on_retry_callback=effective_config.on_retry_callback,
                )
            
            last_exception = None
            
            for attempt in range(1, effective_config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except effective_config.non_retryable_exceptions as e:
                    # Don't retry these
                    logger.warning(
                        f"Non-retryable exception in {func.__name__}: {type(e).__name__}: {e}"
                    )
                    raise
                except effective_config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < effective_config.max_attempts:
                        delay = calculate_delay(attempt, effective_config)
                        
                        logger.warning(
                            f"Retry {attempt}/{effective_config.max_attempts} for {func.__name__} "
                            f"after {type(e).__name__}: {e}. Waiting {delay:.1f}s..."
                        )
                        
                        # Call retry callback if provided
                        if effective_config.on_retry_callback:
                            effective_config.on_retry_callback(attempt, e, delay)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {effective_config.max_attempts} retry attempts failed for {func.__name__}"
                        )
                except Exception as e:
                    # Unexpected exception - don't retry
                    logger.error(
                        f"Unexpected exception in {func.__name__}: {type(e).__name__}: {e}\n"
                        f"Stack trace: {traceback.format_exc()}"
                    )
                    raise
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


# =============================================================================
# Error Context Manager
# =============================================================================

class ErrorContext:
    """
    Context manager for enhanced error handling and logging.
    
    Usage:
        with ErrorContext("Training model", solution_id="sol_123"):
            train_model()
    """
    
    def __init__(
        self,
        operation: str,
        error_class: Type[PlexeError] = PlexeError,
        reraise: bool = True,
        log_level: int = logging.ERROR,
        **context
    ):
        self.operation = operation
        self.error_class = error_class
        self.reraise = reraise
        self.log_level = log_level
        self.context = context
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # Log with full context
            error_msg = (
                f"Error during '{self.operation}': {type(exc_val).__name__}: {exc_val}\n"
                f"Context: {self.context}\n"
                f"Stack trace:\n{traceback.format_exc()}"
            )
            logger.log(self.log_level, error_msg)
            
            if self.reraise:
                if isinstance(exc_val, PlexeError):
                    # Already a PlexeError, just re-raise
                    return False
                else:
                    # Wrap in appropriate PlexeError
                    raise self.error_class(
                        message=f"Error during '{self.operation}': {str(exc_val)}",
                        cause=exc_val,
                        context=self.context,
                    ) from exc_val
            else:
                return True  # Suppress exception
        
        return False


# =============================================================================
# Utility Functions
# =============================================================================

def format_exception_chain(exc: Exception) -> str:
    """
    Format an exception and its full cause chain as a readable string.
    """
    lines = []
    current = exc
    depth = 0
    
    while current is not None:
        prefix = "  " * depth
        exc_type = type(current).__name__
        lines.append(f"{prefix}{'└─ ' if depth > 0 else ''}{exc_type}: {str(current)}")
        
        if hasattr(current, '__traceback__') and current.__traceback__:
            tb_lines = traceback.format_tb(current.__traceback__)
            for tb_line in tb_lines[-3:]:  # Last 3 frames
                for sub_line in tb_line.strip().split('\n'):
                    lines.append(f"{prefix}   {sub_line}")
        
        current = getattr(current, '__cause__', None) or getattr(current, 'cause', None)
        depth += 1
    
    return "\n".join(lines)


def log_exception(
    exc: Exception,
    message: str = "An error occurred",
    level: int = logging.ERROR,
    include_trace: bool = True,
) -> None:
    """
    Log an exception with full context.
    """
    log_parts = [f"{message}: {type(exc).__name__}: {str(exc)}"]
    
    if isinstance(exc, PlexeError) and exc.context:
        log_parts.append(f"Context: {exc.context}")
    
    if include_trace:
        log_parts.append(f"Stack trace:\n{traceback.format_exc()}")
    
    logger.log(level, "\n".join(log_parts))


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    error_class: Type[PlexeError] = PlexeError,
    error_message: str = "Operation failed",
    reraise: bool = False,
    **kwargs,
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for func
        default: Default value to return on error
        error_class: Exception class to wrap errors in
        error_message: Message for wrapped exception
        reraise: Whether to reraise the exception
        **kwargs: Keyword arguments for func
    
    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_exception(e, error_message, include_trace=True)
        
        if reraise:
            if isinstance(e, PlexeError):
                raise
            raise error_class(error_message, cause=e) from e
        
        return default


def create_error_result(
    error: Exception,
    operation: str,
    include_trace: bool = True,
) -> Dict[str, Any]:
    """
    Create a standardized error result dictionary.
    
    This replaces the pattern of returning empty dict/list on error,
    instead returning a structured error response.
    """
    result = {
        "success": False,
        "error": True,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "operation": operation,
    }
    
    if isinstance(error, PlexeError):
        result["context"] = error.context
        result["recoverable"] = error.recoverable
        if include_trace:
            result["stack_trace"] = error.get_full_trace()
    elif include_trace:
        result["stack_trace"] = traceback.format_exc()
    
    return result
