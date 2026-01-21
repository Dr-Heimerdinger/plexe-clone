"""
Tools for dataset manipulation, splitting, and registration.

These tools help with dataset operations within the model generation pipeline, including
splitting datasets into training, validation, and test sets, registering datasets with
the dataset registry, creating sample data for validation, and previewing dataset content.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from smolagents import tool

from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.core.object_registry import ObjectRegistry
from plexe.internal.models.entities.code import Code

logger = logging.getLogger(__name__)


@tool
def register_split_datasets(
    dataset_name: str,
    train_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    splitting_code: str,
) -> Dict[str, str]:
    """
    Register train, validation, and test datasets in the object registry after custom splitting.
    This tool allows the agent to register datasets after performing custom splitting logic.

    Args:
        dataset_name: Original name of the dataset that was split
        train_dataset: pandas DataFrame containing training data
        validation_dataset: pandas DataFrame containing validation data
        test_dataset: pandas DataFrame containing test data
        splitting_code: the code that was used to split the dataset

    Returns:
        Dictionary containing lists of registered dataset names:
        {
            "train_dataset": name of the training dataset,
            "validation_dataset": name of the validation dataset,
            "test_dataset": name of the test dataset,
            "dataset_size": Dictionary with sizes of each dataset
        }
    """

    # Initialize the dataset registry
    object_registry = ObjectRegistry()

    # Initialize the dataset sizes dictionary
    dataset_sizes = {"train": [], "validation": [], "test": []}

    # Register each split dataset
    # Convert pandas DataFrames to TabularDataset objects
    train_ds = DatasetAdapter.coerce(train_dataset)
    val_ds = DatasetAdapter.coerce(validation_dataset)
    test_ds = DatasetAdapter.coerce(test_dataset)

    # Register split datasets in the registry
    train_name = f"{dataset_name}_train"
    val_name = f"{dataset_name}_val"
    test_name = f"{dataset_name}_test"

    object_registry.register(TabularConvertible, train_name, train_ds, overwrite=True, immutable=True)
    object_registry.register(TabularConvertible, val_name, val_ds, overwrite=True, immutable=True)
    object_registry.register(TabularConvertible, test_name, test_ds, overwrite=True, immutable=True)
    object_registry.register(Code, "dataset_splitting_code", Code(splitting_code), overwrite=True)

    # Store dataset sizes
    dataset_sizes["train"].append(len(train_ds))
    dataset_sizes["validation"].append(len(val_ds))
    dataset_sizes["test"].append(len(test_ds))

    logger.debug(
        f"✅ Registered custom split of dataset {dataset_name} into train/validation/test with sizes "
        f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
    )

    return {
        "training_dataset": train_name,
        "validation_dataset": val_name,
        "test_dataset": test_name,
        "dataset_size": dataset_sizes,
    }


@tool
def export_datasets(
    output_dir: str,
    format: str = "csv",
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Export registered train, validation, and test datasets to files (CSV or Parquet).
    This tool allows agents to persist datasets to disk for external analysis, reproducibility,
    or manual inspection by users.

    Args:
        output_dir: Directory path where datasets will be saved (relative to session workdir or absolute).
        format: Output format - 'csv', 'parquet', or 'both'. Defaults to 'csv'.
        include_metadata: Whether to save a metadata JSON file with dataset info. Defaults to True.

    Returns:
        Dictionary containing:
        - exported_files: List of paths to exported files
        - dataset_stats: Statistics for each exported dataset (rows, columns, label distribution if present)
        - metadata_file: Path to metadata file (if include_metadata=True)
    """
    import json
    import os
    from datetime import datetime
    from pathlib import Path

    object_registry = ObjectRegistry()

    # Determine output directory
    try:
        base_workdir = object_registry.get(str, "working_dir")
    except KeyError:
        try:
            session_id = object_registry.get(str, "session_id")
            base_workdir = os.path.join(".workdir", session_id)
        except KeyError:
            base_workdir = ".workdir"

    if os.path.isabs(output_dir):
        export_path = Path(output_dir)
    else:
        export_path = Path(base_workdir) / output_dir

    export_path.mkdir(parents=True, exist_ok=True)

    # Find all train/val/test datasets
    all_datasets = object_registry.list_by_type(TabularConvertible)

    splits_to_export = {}
    for name in all_datasets:
        if name.endswith("_train"):
            splits_to_export.setdefault("train", []).append(name)
        elif name.endswith("_val"):
            splits_to_export.setdefault("validation", []).append(name)
        elif name.endswith("_test"):
            splits_to_export.setdefault("test", []).append(name)

    if not splits_to_export:
        raise ValueError(
            "No train/val/test datasets found in registry. "
            "Ensure datasets have been split using register_split_datasets first."
        )

    exported_files = []
    dataset_stats = {}

    for split_type, dataset_names in splits_to_export.items():
        # Prefer transformed versions if available
        transformed = [n for n in dataset_names if "_transformed_" in n]
        chosen_name = transformed[0] if transformed else dataset_names[0]

        try:
            dataset = object_registry.get(TabularConvertible, chosen_name)
            df = dataset.to_pandas()

            # Calculate stats
            stats = {
                "dataset_name": chosen_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
            }

            # Check for label column and compute distribution
            label_cols = [c for c in df.columns if c.lower() in ("label", "target", "y")]
            if label_cols:
                label_col = label_cols[0]
                value_counts = df[label_col].value_counts().to_dict()
                stats["label_column"] = label_col
                stats["label_distribution"] = {str(k): int(v) for k, v in value_counts.items()}

            dataset_stats[split_type] = stats

            # Export based on format
            base_filename = f"{split_type}"
            if format in ("csv", "both"):
                csv_path = export_path / f"{base_filename}.csv"
                df.to_csv(csv_path, index=False)
                exported_files.append(str(csv_path))
                logger.info(f"✅ Exported {split_type} dataset to {csv_path}")

            if format in ("parquet", "both"):
                parquet_path = export_path / f"{base_filename}.parquet"
                df.to_parquet(parquet_path, index=False)
                exported_files.append(str(parquet_path))
                logger.info(f"✅ Exported {split_type} dataset to {parquet_path}")


        except Exception as e:
            logger.warning(f"⚠️ Failed to export {split_type} dataset ({chosen_name}): {str(e)}")
            dataset_stats[split_type] = {"error": str(e)}

    result = {
        "exported_files": exported_files,
        "dataset_stats": dataset_stats,
        "output_directory": str(export_path),
    }

    # Save metadata
    if include_metadata and exported_files:
        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "format": format,
            "datasets": dataset_stats,
        }

        # Try to include temporal split info if available
        try:
            temporal_info = object_registry.get(dict, "temporal_split_info")
            metadata["temporal_split_info"] = temporal_info
        except KeyError:
            pass

        metadata_path = export_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        result["metadata_file"] = str(metadata_path)
        logger.info(f"✅ Saved metadata to {metadata_path}")

    return result


# TODO: does not need to be a tool
@tool
def create_input_sample(n_samples: int = 5) -> bool:
    """
    Create and register a synthetic sample input dataset that matches the model's input schema.
    This sample is used for validating inference code.

    Args:
        n_samples: Number of samples to generate (default: 5)

    Returns:
        True if sample was successfully created and registered, False otherwise
    """
    object_registry = ObjectRegistry()
    input_schema = object_registry.get(dict, "input_schema")

    try:
        # Create synthetic sample data that matches the schema
        input_sample_dicts = []

        # Generate synthetic examples
        for i in range(n_samples):
            sample = {}
            for field_name, field_type in input_schema.items():
                # Generate appropriate sample values based on type
                if field_type == "int":
                    sample[field_name] = i * 10
                elif field_type == "float":
                    sample[field_name] = i * 10.5
                elif field_type == "bool":
                    sample[field_name] = i % 2 == 0
                elif field_type == "str":
                    sample[field_name] = f"sample_{field_name}_{i}"
                elif field_type == "List[int]":
                    sample[field_name] = [i * 10, i * 20, i * 30]
                elif field_type == "List[float]":
                    sample[field_name] = [i * 10.5, i * 20.5, i * 30.5]
                elif field_type == "List[bool]":
                    sample[field_name] = [True, False, i % 2 == 0]
                elif field_type == "List[str]":
                    sample[field_name] = [f"item_{i}_1", f"item_{i}_2", f"item_{i}_3"]
                else:
                    sample[field_name] = None
            input_sample_dicts.append(sample)

        # TODO: we should use an LLM call to generate sensible values; then validate using pydantic

        # Register the input sample in the registry for validation tool to use
        object_registry.register(list, "predictor_input_sample", input_sample_dicts, overwrite=True, immutable=True)
        logger.debug(
            f"✅ Registered synthetic input sample with {len(input_sample_dicts)} examples for inference validation"
        )
        return True

    except Exception as e:
        logger.warning(f"⚠️ Error creating input sample for validation: {str(e)}")
        return False


@tool
def drop_null_columns(dataset_name: str, keep_columns: List[str] = None) -> Dict[str, Any]:
    """
    Drop all columns from the dataset that are completely null and register the modified dataset.
    
    This tool automatically protects common label column names (label, target, y, class) 
    from being dropped, even if they appear to be null or constant.

    Args:
        dataset_name: Name of the dataset to modify
        keep_columns: Optional list of column names to strictly preserve regardless of their quality.

    Returns:
        Dictionary containing results of the operation:
        - dataset_name: Name of the modified dataset
        - n_dropped: Number of columns dropped
        - dropped_columns: List of names of dropped columns
    """
    object_registry = ObjectRegistry()
    
    # Protected columns that should NEVER be dropped automatically
    PROTECTED_COLUMNS = {'label', 'target', 'y', 'class', 'id', 'user_id', 'item_id'}

    try:
        # Get dataset from registry
        dataset = object_registry.get(TabularConvertible, dataset_name)
        df = dataset.to_pandas()
        
        # Normalize protected columns
        if keep_columns:
            protected = set(c.lower() for c in keep_columns) | PROTECTED_COLUMNS
        else:
            protected = PROTECTED_COLUMNS

        # Drop columns with all null values TODO: make this more intelligent
        # Drop columns with >=50% missing values
        null_columns = df.columns[df.isnull().mean() >= 0.5]

        # Drop constant columns (zero variance)
        constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) == 1]

        # Drop quasi-constant columns (e.g., one value in >95% of rows)
        quasi_constant_columns = [
            col for col in df.columns if (df[col].value_counts(dropna=False, normalize=True).values[0] > 0.95)
        ]

        # Drop columns with all unique values (likely IDs) - BUT keep if it looks like a primary ID
        # We'll rely on PROTECTED_COLUMNS for IDs
        unique_columns = [col for col in df.columns if df[col].nunique(dropna=False) == len(df)]

        # Drop duplicate columns
        duplicate_columns = []
        seen = {}
        for col in df.columns:
            col_data = df[col].to_numpy()
            key = col_data.tobytes() if hasattr(col_data, "tobytes") else tuple(col_data)
            if key in seen:
                duplicate_columns.append(col)
            else:
                seen[key] = col

        # Combine all columns to drop (set to avoid duplicates)
        candidates_to_drop = (
            set(null_columns)
            | set(constant_columns)
            | set(quasi_constant_columns)
            | set(unique_columns)
            | set(duplicate_columns)
        )
        
        # Filter out protected columns
        final_drop_list = []
        for col in candidates_to_drop:
            if col.lower() not in protected:
                final_drop_list.append(col)
            else:
                logger.info(f"Protected column '{col}' from being dropped despite meeting drop criteria.")
        
        n_dropped = len(final_drop_list)
        if n_dropped > 0:
            df.drop(columns=final_drop_list, inplace=True)
            
            # Update registry
            new_dataset = DatasetAdapter.coerce(df)
            object_registry.register(TabularConvertible, dataset_name, new_dataset, overwrite=True)
            
        return {
            "dataset_name": dataset_name,
            "n_dropped": n_dropped,
            "dropped_columns": final_drop_list
        }

    except Exception as e:
        logger.error(f"Error dropping null columns: {e}")
        return {"error": str(e)}

        # Unregister the original dataset
        object_registry.delete(TabularConvertible, dataset_name)

        # Register the modified dataset
        object_registry.register(TabularConvertible, dataset_name, DatasetAdapter.coerce(df), immutable=True)

        return f"Successfully dropped {n_dropped} null columns from dataset '{dataset_name}'"

    except Exception as e:
        raise RuntimeError(f"Failed to drop null columns from dataset '{dataset_name}': {str(e)}")


def _generate_preview_from_dataframe(dataset_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a preview dictionary from a pandas DataFrame.
    
    Args:
        dataset_name: Name to use for the dataset
        df: The DataFrame to generate preview from
        
    Returns:
        Dictionary containing dataset preview information
    """
    # Basic shape and data types
    result = {
        "dataset_name": dataset_name,
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_rows": df.head(5).to_dict(orient="records"),
    }

    # Basic statistics
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().to_dict()
        result["summary_stats"] = {
            col: {
                "mean": stats[col].get("mean"),
                "std": stats[col].get("std"),
                "min": stats[col].get("min"),
                "25%": stats[col].get("25%"),
                "median": stats[col].get("50%"),
                "75%": stats[col].get("75%"),
                "max": stats[col].get("max"),
            }
            for col in numeric_cols
        }

    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    result["missing_values"] = {col: count for col, count in missing_counts.items() if count > 0}

    return result


# Key for unified schema cache (shared with graph_processing.py)
UNIFIED_SCHEMA_CACHE_KEY = "unified_db_schema"


def _get_db_connection_string(object_registry: ObjectRegistry) -> str:
    """
    Get the database connection string from the object registry.
    
    Checks multiple sources in order:
    1. Direct db_connection_string registration
    2. Unified schema cache (_connection_string field)
    
    Returns:
        Database connection string if found, None otherwise
    """
    # First try direct registration
    try:
        db_connection = object_registry.get(str, "db_connection_string")
        if db_connection:
            return db_connection
    except KeyError:
        pass
    
    # Try unified schema cache
    try:
        unified = object_registry.get(dict, UNIFIED_SCHEMA_CACHE_KEY)
        if unified and "_connection_string" in unified:
            return unified["_connection_string"]
    except KeyError:
        pass
    
    return None


def try_load_from_database(dataset_name: str, object_registry: ObjectRegistry, limit: int = 1000) -> pd.DataFrame:
    """
    Try to load a table from the database if a connection string is available.
    
    Args:
        dataset_name: Name of the table to load
        object_registry: The object registry instance
        limit: Maximum number of rows to load (default 1000). Set to None for all rows.
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        db_connection = _get_db_connection_string(object_registry)
        if not db_connection:
            return None
            
        from sqlalchemy import create_engine, text
        
        engine = create_engine(db_connection)
        
        # Query a sample of the table (limit for preview performance)
        # First get total row count
        with engine.connect() as conn:
            count_result = conn.execute(text(f'SELECT COUNT(*) FROM "{dataset_name}"')).fetchone()
            total_rows = count_result[0] if count_result else 0
            
            # Load sample for preview
            query = f'SELECT * FROM "{dataset_name}"'
            if limit is not None:
                query += f' LIMIT {limit}'
                
            df = pd.read_sql(text(query), conn)
            
            # Store total row count as metadata
            df.attrs['total_rows'] = total_rows
            
        logger.info(f"Loaded table '{dataset_name}' from database ({len(df)} rows, {total_rows} total)")
        return df
        
    except Exception as e:
        logger.debug(f"Could not load table '{dataset_name}' from database: {e}")
        return None


@tool
def get_dataset_preview(dataset_name: str) -> Dict[str, Any]:
    """
    Generate a concise preview of a dataset with statistical information to help agents understand the data.
    
    This tool supports both:
    1. Datasets registered in the object registry (TabularConvertible)
    2. Database tables when a db_connection_string is available

    Args:
        dataset_name: Name of the dataset or database table to preview

    Returns:
        Dictionary containing dataset information:
        - shape: dimensions of the dataset
        - dtypes: data types of columns
        - summary_stats: basic statistics (mean, median, min/max)
        - missing_values: count of missing values per column
        - sample_rows: sample of the data (5 rows)
        - source: 'registry' or 'database' indicating where data was loaded from
    """
    object_registry = ObjectRegistry()

    try:
        # First try to get dataset from registry
        dataset = object_registry.get(TabularConvertible, dataset_name)
        df = dataset.to_pandas()
        result = _generate_preview_from_dataframe(dataset_name, df)
        result["source"] = "registry"
        return result

    except KeyError:
        # Dataset not in registry - try loading from database
        logger.debug(f"Dataset '{dataset_name}' not in registry, trying database...")
        df = try_load_from_database(dataset_name, object_registry, limit=1000)
        
        if df is not None:
            result = _generate_preview_from_dataframe(dataset_name, df)
            result["source"] = "database"
            
            # If we got total row count metadata, update the shape
            if hasattr(df, 'attrs') and 'total_rows' in df.attrs:
                result["shape"]["total_rows"] = df.attrs['total_rows']
                result["shape"]["rows"] = df.attrs['total_rows']  # Use actual total
                result["shape"]["sample_rows"] = len(df)
                
            # Also register the dataset in the registry for future use
            try:
                object_registry.register(
                    TabularConvertible, 
                    dataset_name, 
                    DatasetAdapter.coerce(df), 
                    overwrite=True, 
                    immutable=True
                )
                logger.info(f"Registered database table '{dataset_name}' as TabularConvertible for future access")
            except Exception as reg_error:
                logger.debug(f"Could not register table in registry: {reg_error}")
                
            return result
        else:
            # Neither in registry nor loadable from database
            logger.warning(f"⚠️ Dataset '{dataset_name}' not found in registry or database")
            return {
                "error": f"Dataset '{dataset_name}' not found. It is neither registered in the object registry nor available as a database table.",
                "dataset_name": dataset_name,
                "hint": "Use get_latest_datasets() to see available registered datasets, or check that the database connection is configured and the table name is correct."
            }

    except Exception as e:
        logger.warning(f"⚠️ Error creating dataset preview: {str(e)}")
        return {
            "error": f"Failed to generate preview for dataset '{dataset_name}': {str(e)}",
            "dataset_name": dataset_name,
        }


@tool
def get_latest_datasets() -> Dict[str, Any]:
    """
    Get the most recent version of each dataset in the pipeline. Automatically detects transformed
    versions and returns the latest. Use this tool to recall what datasets are available.
    
    This tool supports both:
    1. Datasets registered in the object registry (TabularConvertible)
    2. Database tables when a db_connection_string is available

    Returns:
        Dictionary containing:
        - "raw": The original dataset (if available)
        - "transformed": The transformed dataset (if available)
        - "train": Training split (transformed version if available)
        - "val": Validation split (transformed version if available)
        - "test": Test split (transformed version if available)
        - "database_tables": List of available database tables (if db connection exists)
    """
    object_registry = ObjectRegistry()

    try:
        result = {}
        
        # Get datasets from registry
        all_datasets = object_registry.list_by_type(TabularConvertible)
        
        if all_datasets:
            # Find raw datasets (no suffixes)
            raw_datasets = [
                d for d in all_datasets if not any(suffix in d for suffix in ["_train", "_val", "_test", "_transformed"])
            ]
            if raw_datasets:
                # Use the first one (could be enhanced to handle multiple)
                result["raw"] = raw_datasets[0]

            # Find transformed dataset (not a split)
            transformed = [
                d
                for d in all_datasets
                if d.endswith("_transformed")
                and not any(d.endswith(f"_transformed_{split}") for split in ["train", "val", "test"])
            ]
            if transformed:
                result["transformed"] = transformed[0]

            # Find splits - prefer transformed versions
            for split in ["train", "val", "test"]:
                # First look for transformed split
                transformed_split = [d for d in all_datasets if d.endswith(f"_transformed_{split}")]
                if transformed_split:
                    result[split] = transformed_split[0]
                    continue

                # Fall back to regular split
                regular_split = [d for d in all_datasets if d.endswith(f"_{split}") and "_transformed_" not in d]
                if regular_split:
                    result[split] = regular_split[0]
        
        # Check for database tables
        db_connection = _get_db_connection_string(object_registry)
        if db_connection:
            try:
                from sqlalchemy import create_engine, inspect
                
                engine = create_engine(db_connection)
                inspector = inspect(engine)
                table_names = inspector.get_table_names()
                
                if table_names:
                    result["database_tables"] = table_names
                    result["source"] = "database"
                    logger.debug(f"Found {len(table_names)} database tables: {table_names}")
            except Exception as e:
                logger.debug(f"Could not list database tables: {e}")

        return result

    except Exception as e:
        logger.warning(f"⚠️ Error getting latest datasets: {str(e)}")
        return {}


@tool
def get_training_datasets() -> Dict[str, str]:
    """
    Get datasets ready for model training.
    Automatically finds the best available train/validation datasets.

    Returns:
        Dictionary with 'train' and 'validation' dataset names

    Raises:
        ValueError: If training datasets are not found
    """
    object_registry = ObjectRegistry()

    # Check for temporal split info first (RDL case)
    try:
        temporal_info = object_registry.get(dict, "temporal_split_info")
        # If we have temporal split info, we can assume the datasets are conceptually available
        # even if not registered as TabularConvertible (since they might be in DB)
        return {"train": "temporal_train", "validation": "temporal_val"}  # Convention for RDL
    except KeyError:
        pass  # Not an RDL task or split not done yet

    try:
        all_datasets = object_registry.list_by_type(TabularConvertible)

        # Look for train/val pairs, preferring transformed versions
        train_datasets = []
        val_datasets = []

        # First try to find transformed splits
        for d in all_datasets:
            if d.endswith("_transformed_train"):
                train_datasets.append((d, 1))  # Priority 1 for transformed
            elif d.endswith("_train") and "_transformed_" not in d:
                train_datasets.append((d, 2))  # Priority 2 for regular
            elif d.endswith("_transformed_val"):
                val_datasets.append((d, 1))
            elif d.endswith("_val") and "_transformed_" not in d:
                val_datasets.append((d, 2))

        # Sort by priority (lower is better)
        train_datasets.sort(key=lambda x: x[1])
        val_datasets.sort(key=lambda x: x[1])

        if not train_datasets or not val_datasets:
            raise ValueError("Training datasets not found. Ensure datasets have been split into train/validation sets.")

        # Return the best available pair
        return {"train": train_datasets[0][0], "validation": val_datasets[0][0]}

    except ValueError:
        # Re-raise ValueError as is
        raise
    except Exception as e:
        logger.warning(f"⚠️ Error getting training datasets: {str(e)}")
        raise ValueError(f"Failed to get training datasets: {str(e)}")


@tool
def get_test_dataset() -> str:
    """
    Get the name of the test dataset for final model evaluation.

    Returns:
        Name of the test dataset

    Raises:
        ValueError: If test dataset is not found
    """
    object_registry = ObjectRegistry()

    try:
        all_datasets = object_registry.list_by_type(TabularConvertible)

        # Look for test datasets, preferring transformed version
        test_datasets = []

        for d in all_datasets:
            if d.endswith("_transformed_test"):
                test_datasets.append((d, 1))  # Priority 1 for transformed
            elif d.endswith("_test") and "_transformed_" not in d:
                test_datasets.append((d, 2))  # Priority 2 for regular

        if not test_datasets:
            raise ValueError("Test dataset not found. Ensure datasets have been split into train/validation/test sets.")

        # Sort by priority and return the best
        test_datasets.sort(key=lambda x: x[1])
        return test_datasets[0][0]

    except ValueError:
        # Re-raise ValueError as is
        raise
    except Exception as e:
        logger.warning(f"⚠️ Error getting test dataset: {str(e)}")
        raise ValueError(f"Failed to get test dataset: {str(e)}")


# TODO: this can return a very large amount of data, consider dividing this into list_reports() and get_report(name)
@tool
def get_dataset_reports() -> Dict[str, Dict]:
    """
    Get all available data analysis reports, including EDA for raw datasets and feature engineering reports
    for transformed datasets.

    Returns:
        Dictionary with the following structure:

    """
    object_registry = ObjectRegistry()

    try:
        # Get all dict objects from registry
        all_dicts = object_registry.list_by_type(dict)

        # Filter for EDA reports (they have pattern "eda_report_{dataset_name}")
        eda_reports = {}
        for name in all_dicts:
            if name.startswith("eda_report_"):
                # Extract dataset name
                dataset_name = name[11:]  # Remove "eda_report_" prefix
                try:
                    report = object_registry.get(dict, name)
                    eda_reports[dataset_name] = report
                except Exception as e:
                    logger.debug(f"Failed to retrieve EDA report {name}: {str(e)}")
                    continue

        # Filter for feature engineering reports (they have pattern "fe_report_{dataset_name}")
        fe_reports = {}
        for name in all_dicts:
            if name.startswith("fe_report_"):
                # Extract dataset name
                dataset_name = name[10:]  # Remove "fe_report_" prefix
                try:
                    report = object_registry.get(dict, name)
                    fe_reports[dataset_name] = report
                except Exception as e:
                    logger.debug(f"Failed to retrieve Feature Engineering report {name}: {str(e)}")
                    continue

        return {
            "eda_reports": eda_reports,
            "feature_engineering_reports": fe_reports,
        }

    except Exception as e:
        logger.warning(f"⚠️ Error getting EDA reports: {str(e)}")
        return {}


@tool
def register_data_source_path(path: str, description: str = "raw_csv_files") -> str:
    """
    Register a file system path (directory or file) in the ObjectRegistry so other agents can locate data.
    
    Args:
        path: The absolute or relative path to the data.
        description: A short key/description for this path (default: "raw_csv_files").
                     This will be used as the key in the registry (e.g., "path_raw_csv_files").
    
    Returns:
        Confirmation message.
    """
    try:
        object_registry = ObjectRegistry()
        key = f"path_{description}"
        object_registry.register(str, key, path, overwrite=True)
        logger.info(f"✅ Registered data path '{path}' under key '{key}'")
        return f"Successfully registered path '{path}' under key '{key}'"
    except Exception as e:
        logger.error(f"Failed to register data path: {e}")
        return f"Error registering data path: {str(e)}"
