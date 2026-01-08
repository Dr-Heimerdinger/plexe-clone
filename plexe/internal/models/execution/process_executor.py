"""
Module: ProcessExecutor for Isolated Python Code Execution

This module provides an implementation of the `Executor` interface for executing Python code snippets
in an isolated process. It captures stdout, stderr, exceptions, and stack traces, and enforces
timeout limits on execution.

Classes:
    - RedirectQueue: A helper class to redirect stdout and stderr to a multiprocessing Queue.
    - ProcessExecutor: A class to execute Python code snippets in an isolated process.

Usage:
    Create an instance of `ProcessExecutor`, providing the Python code, working directory, and timeout.
    Call the `run` method to execute the code and return the results in an `ExecutionResult` object.

Exceptions:
    - Raises `RuntimeError` if the child process fails unexpectedly.

"""

import logging
import subprocess
import sys
import time
import traceback
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from typing import Dict, Optional

from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.utils.response import extract_performance
from plexe.internal.models.execution.executor import ExecutionResult, Executor
from plexe.internal.common.errors import CodeExecutionError
from plexe.config import config

logger = logging.getLogger(__name__)


class ProcessExecutor(Executor):
    """
    Execute Python code snippets in an isolated process.

    The `ProcessExecutor` class implements the `Executor` interface, allowing Python code
    snippets to be executed with strict isolation, output capture, and timeout enforcement.
    
    Features:
    - Retry logic for transient failures (file I/O, process spawning)
    - Detailed error tracking with stack traces
    - Proper resource cleanup
    """

    # Class-level retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 2.0  # seconds, exponential backoff

    def __init__(
        self,
        execution_id: str,
        code: str,
        working_dir: Path | str,
        datasets: Dict[str, TabularConvertible],
        timeout: int,
        code_execution_file_name: str = config.execution.runfile_name,
    ):
        """
        Initialize the ProcessExecutor.

        Args:
            execution_id (str): Unique identifier for this execution.
            code (str): The Python code to execute.
            working_dir (Path | str): The working directory for execution.
            datasets (Dict[str, TabularConvertible]): Datasets to be used for execution.
            timeout (int): The maximum allowed execution time in seconds.
            code_execution_file_name (str): The filename to use for the executed script.
        """
        super().__init__(code, timeout)
        # Create a unique working directory for this execution
        self.working_dir = Path(working_dir).resolve() / execution_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        # Set the file names for the code and training data
        self.code_file_name = code_execution_file_name
        self.datasets = datasets
        # Keep track of resources for cleanup
        self.dataset_files = []
        self.code_file = None
        self.process = None
        self.execution_id = execution_id
        self._last_error: Optional[Exception] = None

    def _prepare_execution_environment(self) -> None:
        """
        Prepare the execution environment by writing code and datasets to files.
        
        Raises:
            CodeExecutionError: If file preparation fails
        """
        try:
            # Write code to file with module environment setup
            self.code_file = self.working_dir / self.code_file_name
            module_setup = "import os\nimport sys\nfrom pathlib import Path\n\n"
            with open(self.code_file, "w", encoding="utf-8") as f:
                f.write(module_setup + self.code)

            # Write datasets to files
            self.dataset_files = []
            for dataset_name, dataset in self.datasets.items():
                dataset_file: Path = self.working_dir / f"{dataset_name}.parquet"
                pq.write_table(pa.Table.from_pandas(df=dataset.to_pandas()), dataset_file)
                self.dataset_files.append(dataset_file)
                
        except Exception as e:
            error_msg = f"Failed to prepare execution environment: {str(e)}"
            logger.error(f"🔥 {error_msg}\nStack trace:\n{traceback.format_exc()}")
            raise CodeExecutionError(
                message=error_msg,
                code=self.code[:500] + "..." if len(self.code) > 500 else self.code,
                cause=e,
                context={"execution_id": self.execution_id, "working_dir": str(self.working_dir)},
            ) from e

    def _execute_subprocess(self, attempt: int = 1) -> ExecutionResult:
        """
        Execute the code in a subprocess (single attempt).
        
        Args:
            attempt: Current attempt number for logging
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(self.code_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.working_dir),
                text=True,
            )

            stdout, stderr = self.process.communicate(timeout=self.timeout)
            exec_time = time.time() - start_time

            # Collect all model artifacts created by the execution
            model_artifacts = self._collect_artifacts()

            if self.process.returncode != 0:
                error_msg = (
                    f"Process exited with code {self.process.returncode} "
                    f"(attempt {attempt}/{self.MAX_RETRIES})"
                )
                logger.warning(
                    f"⚠️ {error_msg}\n"
                    f"Stderr preview: {stderr[:500] if stderr else 'None'}\n"
                    f"Stdout preview: {stdout[:200] if stdout else 'None'}"
                )
                return ExecutionResult(
                    term_out=[stdout],
                    exec_time=exec_time,
                    exception=CodeExecutionError(
                        message=error_msg,
                        code=self.code[:200],
                        stdout=stdout,
                        stderr=stderr,
                        return_code=self.process.returncode,
                        context={"attempt": attempt, "execution_id": self.execution_id},
                    ),
                    model_artifact_paths=model_artifacts,
                )

            logger.debug(f"✅ Code executed successfully in {exec_time:.2f}s (attempt {attempt})")
            return ExecutionResult(
                term_out=[stdout],
                exec_time=exec_time,
                model_artifact_paths=model_artifacts,
                performance=extract_performance(stdout),
            )

        except subprocess.TimeoutExpired:
            if self.process:
                self.process.kill()

            error_msg = f"Execution exceeded {self.timeout}s timeout (attempt {attempt}/{self.MAX_RETRIES})"
            logger.error(f"🔥 {error_msg}")
            return ExecutionResult(
                term_out=[],
                exec_time=self.timeout,
                exception=CodeExecutionError(
                    message=error_msg,
                    code=self.code[:200],
                    context={"timeout": self.timeout, "attempt": attempt},
                ),
            )
            
    def _collect_artifacts(self) -> list:
        """Collect model artifacts from the working directory."""
        model_artifacts = []
        model_dir = self.working_dir / "model_files"
        if model_dir.exists() and model_dir.is_dir():
            model_artifacts.append(str(model_dir))
        else:
            for file in self.working_dir.iterdir():
                if file != self.code_file and file not in self.dataset_files:
                    model_artifacts.append(str(file))
        return model_artifacts

    def run(self) -> ExecutionResult:
        """
        Execute code in a subprocess with retry logic for transient failures.
        
        Returns:
            ExecutionResult with execution details, performance metrics, and any errors
        """
        logger.debug(f"ProcessExecutor starting execution in: {self.working_dir}")
        
        # Prepare environment (with its own error handling)
        self._prepare_execution_environment()
        
        last_result = None
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                result = self._execute_subprocess(attempt)
                last_result = result
                
                # Success or non-retryable error
                if result.exception is None:
                    return result
                
                # Check if error is retryable (process spawn failures, file I/O issues)
                if isinstance(result.exception, CodeExecutionError):
                    exc = result.exception
                    # Timeout and process errors are generally not retryable
                    if "timeout" in str(exc).lower():
                        logger.error(f"❌ Timeout error - not retrying")
                        return result
                    
                    # Non-zero return codes might be code bugs, not infrastructure issues
                    if exc.return_code is not None and exc.return_code != 0:
                        # Check stderr for transient errors
                        if exc.stderr and any(
                            transient in exc.stderr.lower() 
                            for transient in ["memory", "resource", "temporary", "busy"]
                        ):
                            logger.info(f"🔄 Transient error detected, will retry...")
                        else:
                            logger.error(f"❌ Code error (return code {exc.return_code}) - not retrying")
                            return result
                
                # Retry with backoff
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY_BASE * (2 ** (attempt - 1))
                    logger.info(f"⏳ Retrying in {delay:.1f}s... (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    time.sleep(delay)
                    
            except Exception as e:
                # Unexpected error during execution
                error_msg = f"Unexpected error during execution: {type(e).__name__}: {e}"
                logger.error(f"🔥 {error_msg}\nStack trace:\n{traceback.format_exc()}")
                
                if attempt >= self.MAX_RETRIES:
                    return ExecutionResult(
                        term_out=[f"Execution failed: {error_msg}"],
                        exec_time=0,
                        exception=CodeExecutionError(
                            message=error_msg,
                            cause=e,
                            context={
                                "attempt": attempt,
                                "execution_id": self.execution_id,
                                "stack_trace": traceback.format_exc(),
                            },
                        ),
                    )
                
                delay = self.RETRY_DELAY_BASE * (2 ** (attempt - 1))
                logger.info(f"⏳ Retrying after unexpected error in {delay:.1f}s...")
                time.sleep(delay)
        
        # All retries exhausted
        logger.error(
            f"❌ Execution failed after {self.MAX_RETRIES} attempts.\n"
            f"Last error: {last_result.exception if last_result else 'Unknown'}"
        )
        
        # Cleanup before returning
        self.cleanup()
        return last_result

    def cleanup(self):
        """
        Clean up resources after execution while preserving model artifacts.
        
        This method is idempotent and safe to call multiple times.
        """
        logger.debug(f"Cleaning up resources for execution in {self.working_dir}")

        errors = []

        # Clean up dataset files
        for dataset_file in self.dataset_files:
            try:
                if dataset_file.exists():
                    dataset_file.unlink()
            except Exception as e:
                errors.append(f"Failed to delete dataset file {dataset_file}: {e}")

        # Clean up code file
        if self.code_file:
            try:
                if self.code_file.exists():
                    self.code_file.unlink()
            except Exception as e:
                errors.append(f"Failed to delete code file {self.code_file}: {e}")

        # Terminate process if still running
        if self.process and self.process.poll() is None:
            try:
                self.process.kill()
                self.process.wait(timeout=5)  # Wait for process to terminate
            except Exception as e:
                errors.append(f"Failed to kill process: {e}")

        if errors:
            logger.warning(
                f"Errors during resource cleanup for {self.working_dir}:\n" + 
                "\n".join(f"  - {err}" for err in errors)
            )

    def __del__(self):
        """Ensure cleanup happens when the object is garbage collected."""
        try:
            self.cleanup()
        except Exception:
            # Silent failure during garbage collection - detailed logging already done in cleanup()
            pass
