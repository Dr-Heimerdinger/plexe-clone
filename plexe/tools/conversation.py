"""
Tools for conversational model definition and build initiation.

These tools support the conversational agent in helping users define their ML
requirements and starting model builds when ready.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from smolagents import tool
from sqlalchemy import create_engine, inspect

import plexe
from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.provider import ProviderConfig
from plexe.core.object_registry import ObjectRegistry
from plexe.internal.models.callbacks.mlflow import MLFlowCallback

logger = logging.getLogger(__name__)


def _parse_schema(schema: Any) -> Optional[Dict]:
    """
    Parse a schema that may be provided as a dict, JSON string, or None.
    
    Args:
        schema: The schema to parse (dict, JSON string, or None)
        
    Returns:
        Parsed dictionary or None
    """
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, str):
        try:
            parsed = json.loads(schema)
            if isinstance(parsed, dict):
                return parsed
            logger.warning(f"Schema string parsed but is not a dict: {type(parsed)}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse schema string as JSON: {e}")
            return None
    logger.warning(f"Unexpected schema type: {type(schema)}")
    return None


@tool
def validate_dataset_files(file_paths: List[str]) -> Dict[str, Dict]:
    """
    Check if specified file paths can be read as datasets using pandas.

    Args:
        file_paths: List of file paths to validate

    Returns:
        Dictionary mapping file paths to validation results with status, shape, and error info
    """
    results = {}

    for file_path in file_paths:
        result = {"valid": False, "shape": None, "columns": None, "error": None}

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result["error"] = f"File does not exist: {file_path}"
                results[file_path] = result
                continue

            # Determine file type and try to read
            path_obj = Path(file_path)
            file_extension = path_obj.suffix.lower()

            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            elif file_extension in [".parquet", ".pq"]:
                df = pd.read_parquet(file_path)
            else:
                result["error"] = f"Unsupported file format: {file_extension}. Supported formats: .csv, .parquet"
                results[file_path] = result
                continue

            # File successfully read
            result["valid"] = True
            result["dataset_name"] = path_obj.stem
            result["shape"] = df.shape

            # Register the DataFrame in object registry
            ObjectRegistry().register(
                t=TabularConvertible, name=path_obj.stem, item=DatasetAdapter.coerce(df), immutable=True
            )

        except Exception as e:
            result["error"] = str(e)

        results[file_path] = result

    return results


@tool
def validate_db_connection(connection_string: str) -> Dict[str, Any]:
    """
    Validate a database connection string and retrieve schema information.
    Use this to check if the agent can connect to the provided database.

    Args:
        connection_string: Database connection string (e.g., postgresql+psycopg2://user:pass@host:port/dbname)

    Returns:
        Dictionary with connection status and list of tables if successful.
    """
    try:
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        return {
            "valid": True,
            "tables": table_names,
            "message": f"Successfully connected. Found {len(table_names)} tables: {', '.join(table_names[:5])}..."
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "message": f"Failed to connect to database: {str(e)}"
        }


@tool
def initiate_model_build(
    intent: str,
    dataset_file_paths: List[str] = [],
    db_connection_string: Optional[str] = None,
    input_schema: Optional[Any] = None,
    output_schema: Optional[Any] = None,
    n_solutions_to_try: int = 1,
) -> Dict[str, str]:
    """
    Initiate a model build by loading datasets from file paths OR connecting to a database, and starting the build process.

    Args:
        intent: Natural language description of what the model should do
        dataset_file_paths: List of file paths to dataset files (CSV or Parquet). Optional if db_connection_string is provided.
        db_connection_string: Database connection string for Relational Deep Learning. Optional if dataset_file_paths is provided.
        input_schema: The input schema for the model. Can be a flat field:type dictionary or a JSON string. For complex schemas (e.g. graphs), leave None and describe in intent.
        output_schema: The output schema for the model. Can be a flat field:type dictionary or a JSON string. For complex schemas (e.g. graphs), leave None and describe in intent.
        n_solutions_to_try: Number of model solutions to try, out of which the best will be selected

    Returns:
        Dictionary with build initiation status and details
    """
    try:
        # Parse schemas (handle dict, JSON string, or None)
        parsed_input_schema = _parse_schema(input_schema)
        parsed_output_schema = _parse_schema(output_schema)
        
        # Validate inputs
        if not dataset_file_paths and not db_connection_string:
            return {
                "status": "failed",
                "message": "Either dataset_file_paths or db_connection_string must be provided."
            }

        # Validate files if provided
        if dataset_file_paths:
            validation_results = validate_dataset_files(dataset_file_paths)
            failed_files = [path for path, result in validation_results.items() if not result["valid"]]
            if failed_files:
                error_details = {path: validation_results[path]["error"] for path in failed_files}
                return {
                    "status": "failed",
                    "message": f"Failed to read dataset files: {failed_files}",
                    "errors": error_details,
                }

        # Validate DB if provided
        if db_connection_string:
            db_result = validate_db_connection(db_connection_string)
            if not db_result["valid"]:
                return {
                    "status": "failed",
                    "message": f"Invalid database connection: {db_result['error']}"
                }

        # Load datasets into DataFrames (if any)
        datasets = []
        for file_path in dataset_file_paths:
            path_obj = Path(file_path)
            file_extension = path_obj.suffix.lower()

            if file_extension == ".csv":
                df = pd.read_csv(file_path)
                datasets.append(df)
            elif file_extension in [".parquet", ".pq"]:
                df = pd.read_parquet(file_path)
                datasets.append(df)

        # Import here to avoid circular dependencies
        from plexe.model_builder import ModelBuilder

        gemini_model = "gemini/gemini-2.5-pro"

        model_builder = ModelBuilder(
            provider=ProviderConfig(
                default_provider=gemini_model,
                orchestrator_provider=gemini_model,
                research_provider=gemini_model,
                engineer_provider=gemini_model,
                ops_provider=gemini_model,
                tool_provider=gemini_model,
            ),
            working_dir="./workdir/chat-session/",
        )

        # Start the build process
        logger.info(f"Initiating model build with intent: {intent}")
        if dataset_file_paths:
            logger.info(f"Using dataset files: {dataset_file_paths}")
        if db_connection_string:
            logger.info(f"Using database connection: {db_connection_string}")

        # Set up MLFlow callback if tracking URI is configured
        callbacks = []
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            callbacks.append(
                MLFlowCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name=f"chat-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                )
            )

        model = model_builder.build(
            intent=intent,
            datasets=datasets,
            db_connection_string=db_connection_string,
            input_schema=parsed_input_schema,
            output_schema=parsed_output_schema,
            max_iterations=n_solutions_to_try,
            callbacks=callbacks if callbacks else None,
        )

        plexe.save_model(model, "model-from-chat.tar.gz")

        return {
            "status": "initiated",
            "message": f"Model build started successfully with intent: '{intent}'",
            "dataset_files": dataset_file_paths,
            "db_connection": "Provided" if db_connection_string else "None",
        }

    except Exception as e:
        logger.error(f"Failed to initiate model build: {str(e)}")
        return {"status": "failed", "message": f"Failed to start model build: {str(e)}", "error": str(e)}
