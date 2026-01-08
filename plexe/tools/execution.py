"""
Tools related to code execution, including running training code in isolated environments and
applying feature transformations to datasets.

These tools automatically handle model artifact registration through the ArtifactRegistry,
ensuring that artifacts generated during the execution can be retrieved later in the pipeline.
"""

import logging
import uuid
import types
import warnings
import subprocess
import traceback
from typing import Dict, List, Callable, Type

from smolagents import tool

from plexe.callbacks import Callback
from plexe.internal.common.errors import (
    CodeExecutionError,
    RegistryError,
    DatasetError,
    with_retry,
    RetryConfig,
    DEFAULT_CODE_EXECUTION_RETRY,
    ErrorContext,
    log_exception,
    create_error_result,
)
from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.core.object_registry import ObjectRegistry
from plexe.internal.models.entities.code import Code
from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from plexe.core.entities.solution import Solution
from plexe.internal.models.execution.process_executor import ProcessExecutor

logger = logging.getLogger(__name__)


def get_executor_tool(distributed: bool = False) -> Callable:
    """Get the appropriate executor tool based on the distributed flag."""

    @tool
    def execute_training_code(
        solution_id: str,
        code: str,
        working_dir: str,
        dataset_names: List[str],
        timeout: int,
        metric_to_optimise_name: str,
        metric_to_optimise_comparison_method: str,
    ) -> Dict:
        """Executes training code in an isolated environment and updates the Solution object.

        Args:
            solution_id: ID of the Solution object to update with execution results
            code: The code to execute
            working_dir: Directory to use for execution
            dataset_names: List of dataset names to retrieve from the registry
            timeout: Maximum execution time in seconds
            metric_to_optimise_name: The name of the metric to optimize for
            metric_to_optimise_comparison_method: The comparison method for the metric

        Returns:
            A dictionary containing execution results with model artifacts and their registry names
        """
        # Log the distributed flag
        logger.debug(f"execute_training_code called with distributed={distributed}")

        from plexe.callbacks import BuildStateInfo

        object_registry = ObjectRegistry()

        execution_id = f"{solution_id}-{uuid.uuid4()}"
        try:
            # Get the existing Solution object from registry
            solution = object_registry.get(Solution, solution_id)

            # Get actual datasets from registry
            datasets = object_registry.get_multiple(TabularConvertible, dataset_names)

            # Convert string to enum if needed
            if "HIGHER_IS_BETTER" in metric_to_optimise_comparison_method:
                comparison_method = ComparisonMethod.HIGHER_IS_BETTER
            elif "LOWER_IS_BETTER" in metric_to_optimise_comparison_method:
                comparison_method = ComparisonMethod.LOWER_IS_BETTER
            elif "TARGET_IS_BETTER" in metric_to_optimise_comparison_method:
                comparison_method = ComparisonMethod.TARGET_IS_BETTER
            else:
                comparison_method = ComparisonMethod.HIGHER_IS_BETTER

            # Update the solution with training code and get callbacks
            solution.training_code = code
            # Create state info once for all callbacks
            state_info = BuildStateInfo(
                intent="Unknown",  # Will be filled by agent context
                provider="Unknown",  # Will be filled by agent context
                input_schema=None,  # Will be filled by agent context
                output_schema=None,  # Will be filled by agent context
                datasets=datasets,
                iteration=0,  # Default value, no longer used for MLFlow run naming
                node=solution,
            )

            # Notify all callbacks about execution start
            _notify_callbacks(object_registry.get_all(Callback), "start", state_info)

            # Import here to avoid circular imports
            from plexe.config import config

            # Get the appropriate executor class via the factory
            executor_class = _get_executor_class(distributed=distributed)

            # Create an instance of the executor
            logger.debug(f"Creating {executor_class.__name__} for execution ID: {execution_id}")
            executor = executor_class(
                execution_id=execution_id,
                code=code,
                working_dir=working_dir,
                datasets=datasets,
                timeout=timeout,
                code_execution_file_name=config.execution.runfile_name,
            )

            # Execute and collect results - ProcessExecutor.run() handles cleanup internally
            logger.debug(f"Executing solution {solution} using executor {executor}")
            result = executor.run()
            logger.debug(f"Execution result: {result}")
            solution.execution_time = result.exec_time
            solution.execution_stdout = result.term_out
            solution.exception_was_raised = result.exception is not None
            solution.exception = result.exception or None
            solution.model_artifacts = [Artifact.from_path(p) for p in result.model_artifact_paths]

            # Handle the performance metric properly using the consolidated validation logic
            performance_value = None
            is_worst = True

            if result.is_valid_performance():
                performance_value = result.performance
                is_worst = False

            # Create a metric object with proper handling of None or invalid values
            solution.performance = Metric(
                name=metric_to_optimise_name,
                value=performance_value,
                comparator=MetricComparator(comparison_method=comparison_method),
                is_worst=is_worst,
            )

            # Notify callbacks about the execution end with the same state_info
            # The solution reference in state_info automatically reflects the updates to solution
            _notify_callbacks(object_registry.get_all(Callback), "end", state_info)

            # Check if the execution failed in any way
            if solution.exception is not None:
                raise RuntimeError(f"Execution failed with exception: {solution.exception}")
            if not result.is_valid_performance():
                raise RuntimeError(f"Execution failed due to not producing a valid performance: {result.performance}")

            # Register artifacts and update solution in registry
            object_registry.register_multiple(Artifact, {a.name: a for a in solution.model_artifacts})

            # Update the solution in the registry with all execution results
            object_registry.register(Solution, solution_id, solution, overwrite=True)

            # Return results
            return {
                "success": not solution.exception_was_raised,
                "performance": (
                    {
                        "name": solution.performance.name if solution.performance else None,
                        "value": solution.performance.value if solution.performance else None,
                        "comparison_method": (
                            str(solution.performance.comparator.comparison_method) if solution.performance else None
                        ),
                    }
                    if solution.performance
                    else None
                ),
                "exception": str(solution.exception) if solution.exception else None,
                "model_artifact_names": [a.name for a in solution.model_artifacts],
                "solution_id": solution_id,
            }
        except Exception as e:
            # Log full stack trace at debug level
            import traceback

            logger.debug(f"Error executing training code: {str(e)}\n{traceback.format_exc()}")

            return {
                "success": False,
                "performance": None,
                "exception": str(e),
                "model_artifact_names": [],
            }

    return execute_training_code


def _get_executor_class(distributed: bool = False) -> Type:
    """Get the appropriate executor class based on the distributed flag.

    Args:
        distributed: Whether to use distributed execution if available

    Returns:
        Executor class (not instance) appropriate for the environment
    """
    # Log the distributed flag
    logger.debug(f"get_executor_class using distributed={distributed}")
    if distributed:
        try:
            # Try to import Ray executor
            from plexe.internal.models.execution.ray_executor import RayExecutor

            logger.debug("Using Ray for distributed execution")
            return RayExecutor
        except ImportError:
            # Fall back to process executor if Ray is not available
            logger.warning("Ray not available, falling back to ProcessExecutor")
            return ProcessExecutor

    # Default to ProcessExecutor for non-distributed execution
    logger.debug("Using ProcessExecutor (non-distributed)")
    return ProcessExecutor


def _notify_callbacks(callbacks: Dict, event_type: str, build_state_info) -> None:
    """Helper function to notify callbacks with consistent error handling.

    Args:
        callbacks: Dictionary of callbacks from the registry
        event_type: The event type - either "start" or "end"
        build_state_info: The state info to pass to callbacks
    """
    method_name = f"on_iteration_{event_type}"

    for callback in callbacks.values():
        try:
            getattr(callback, method_name)(build_state_info)
        except Exception as e:
            # Log full stack trace at debug level
            import traceback

            logger.debug(
                f"Error in callback {callback.__class__.__name__}.{method_name}: {e}\n{traceback.format_exc()}"
            )
            # Log a shorter message at warning level
            logger.warning(f"Error in callback {callback.__class__.__name__}.{method_name}: {str(e)[:50]}")


@tool
def apply_feature_transformer(dataset_name: str) -> Dict:
    """
    Applies a feature transformer to datasets and registers the transformed datasets. The name of the
    new transformed dataset is returned in the response.

    Args:
        dataset_name: Name of datasets to transform

    Returns:
        Dictionary with results of transformation:
        - success: Boolean indicating success or failure
        - original_dataset_name: Name of the original dataset
        - new_dataset_name: Name of the transformed dataset
    """
    object_registry = ObjectRegistry()

    try:
        # Get feature transformer code from registry
        code_obj = object_registry.get(Code, "feature_transformations")
        transformer_code = code_obj.code

        # Load code as module
        module = types.ModuleType("feature_transformer_module")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if transformer_code and isinstance(transformer_code, str):
                exec(transformer_code, module.__dict__)
            else:
                raise ValueError("Feature transformer code is invalid or missing.")

        # Instantiate transformer
        transformer = module.FeatureTransformerImplementation()

        # Get dataset
        dataset = object_registry.get(TabularConvertible, dataset_name)
        df = dataset.to_pandas()

        # Apply transformation
        transformed_df = transformer.transform(df)

        # Register transformed dataset
        transformed_name = f"{dataset_name}_transformed"
        transformed_ds = DatasetAdapter.coerce(transformed_df)
        object_registry.register(TabularConvertible, transformed_name, transformed_ds, overwrite=True, immutable=True)

        logger.debug(f"✅ Applied feature transformer to {dataset_name} → {transformed_name}")

        return {"success": True, "original_dataset_name": dataset_name, "new_dataset_name": transformed_name}
    except Exception as e:
        error_msg = f"Error applying feature transformer to {dataset_name}: {str(e)}"
        logger.error(f"🔥 {error_msg}\nStack trace:\n{traceback.format_exc()}")
        return {
            "success": False, 
            "error": error_msg,
            "error_type": type(e).__name__,
            "original_dataset_name": dataset_name,
            "stack_trace": traceback.format_exc(),
        }


from plexe.tools.io_manager import write_file


def _execute_code_with_retry(code_file: str, timeout: int, attempt: int = 1, max_attempts: int = 3) -> dict:
    """
    Internal function to execute code with retry logic.
    
    Args:
        code_file: Path to the Python file to execute
        timeout: Timeout in seconds
        attempt: Current attempt number
        max_attempts: Maximum number of retry attempts
    
    Returns:
        Execution result dictionary
    """
    import sys
    
    try:
        process = subprocess.run(
            [sys.executable, code_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        
        result = {
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "returncode": process.returncode,
            "attempt": attempt,
        }
        
        if process.returncode == 0:
            logger.debug(f"✅ Code executed successfully on attempt {attempt}")
        else:
            logger.warning(
                f"⚠️ Code execution failed with return code {process.returncode} "
                f"on attempt {attempt}/{max_attempts}\n"
                f"Stderr: {process.stderr[:500] if process.stderr else 'None'}"
            )
            result["error"] = f"Process exited with code {process.returncode}"
            result["error_type"] = "ExecutionError"
        
        return result
        
    except subprocess.TimeoutExpired as e:
        error_msg = f"Code execution timed out after {timeout}s on attempt {attempt}/{max_attempts}"
        logger.error(f"🔥 {error_msg}")
        return {
            "success": False,
            "stdout": str(e.stdout) if e.stdout else "",
            "stderr": str(e.stderr) if e.stderr else "",
            "returncode": -1,
            "error": error_msg,
            "error_type": "TimeoutExpired",
            "timeout_seconds": timeout,
            "attempt": attempt,
        }
    except OSError as e:
        # OS errors (file not found, permission denied) might be retryable
        error_msg = f"OS error during code execution: {e}"
        logger.error(f"🔥 {error_msg}\nStack trace:\n{traceback.format_exc()}")
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "error": error_msg,
            "error_type": "OSError",
            "attempt": attempt,
            "retryable": True,
            "stack_trace": traceback.format_exc(),
        }
    except Exception as e:
        error_msg = f"Unexpected error during code execution: {type(e).__name__}: {e}"
        logger.error(f"🔥 {error_msg}\nStack trace:\n{traceback.format_exc()}")
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "error": error_msg,
            "error_type": type(e).__name__,
            "attempt": attempt,
            "retryable": False,
            "stack_trace": traceback.format_exc(),
        }


@tool
def execute_code(code: str, timeout: int = 60, max_retries: int = 2) -> dict:
    """
    Executes a string of Python code in a subprocess with retry logic.

    Args:
        code: The Python code to execute.
        timeout: The timeout in seconds for the execution.
        max_retries: Maximum number of retry attempts for retryable errors.

    Returns:
        A dictionary containing:
        - success: Boolean indicating if execution succeeded
        - stdout: Standard output from execution
        - stderr: Standard error from execution  
        - returncode: Process return code
        - error: Error message if failed
        - error_type: Type of error if failed
        - attempt: Which attempt succeeded/failed
        - stack_trace: Full stack trace on error
    """
    import time
    
    # Write code to temporary file
    code_file = write_file("code.py", code)
    
    last_result = None
    
    for attempt in range(1, max_retries + 1):
        result = _execute_code_with_retry(code_file, timeout, attempt, max_retries)
        last_result = result
        
        if result.get("success", False):
            return result
        
        # Check if error is retryable
        if not result.get("retryable", False) or attempt >= max_retries:
            break
        
        # Exponential backoff before retry
        delay = min(2 ** attempt, 10)  # Max 10 seconds
        logger.info(f"⏳ Retrying code execution in {delay}s... (attempt {attempt + 1}/{max_retries})")
        time.sleep(delay)
    
    # All attempts failed
    logger.error(
        f"🔥 Code execution failed after {max_retries} attempts.\n"
        f"Last error: {last_result.get('error', 'Unknown')}\n"
        f"Code preview: {code[:200]}..."
    )
    
    return last_result
