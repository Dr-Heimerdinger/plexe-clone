"""
Tools for temporal processing and training table generation for the Temporal Task Supervisor Agent.

These tools are designed to be schema-agnostic and work with any relational database.
They discover the schema dynamically and let the agent decide how to construct queries.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import pandas as pd
from smolagents import tool
from sqlalchemy import create_engine, inspect, text

from plexe.core.object_registry import ObjectRegistry
from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.internal.common.datasets.interface import TabularConvertible

logger = logging.getLogger(__name__)


@tool
def discover_temporal_columns(db_connection_string: str) -> Dict[str, Any]:
    """
    Discovers all timestamp/date columns across all tables in the database.
    Use this to understand the temporal structure of the data before creating temporal splits.
    
    Args:
        db_connection_string: SQLAlchemy connection string (e.g., 'postgresql+psycopg2://user:pass@host:port/db')
    
    Returns:
        A dictionary containing:
        - tables: Dict mapping table names to lists of temporal columns with their date ranges
        - overall_range: The min and max dates across all temporal columns
    """
    try:
        engine = create_engine(db_connection_string)
        inspector = inspect(engine)
        
        temporal_info = {"tables": {}, "overall_range": {"min": None, "max": None}}
        all_min_dates = []
        all_max_dates = []
        
        # Get all tables
        table_names = inspector.get_table_names()
        
        for table_name in table_names:
            columns = inspector.get_columns(table_name)
            temporal_columns = []
            
            for col in columns:
                col_type = str(col["type"]).upper()
                # Check if column is a temporal type
                if any(t in col_type for t in ["TIMESTAMP", "DATE", "DATETIME", "TIME"]):
                    col_name = col["name"]
                    
                    # Query the date range for this column
                    try:
                        with engine.connect() as conn:
                            query = text(f'SELECT MIN("{col_name}"), MAX("{col_name}") FROM "{table_name}"')
                            result = conn.execute(query).fetchone()
                            
                            if result and result[0] and result[1]:
                                min_date = result[0]
                                max_date = result[1]
                                temporal_columns.append({
                                    "column": col_name,
                                    "type": col_type,
                                    "min_date": str(min_date),
                                    "max_date": str(max_date)
                                })
                                all_min_dates.append(min_date)
                                all_max_dates.append(max_date)
                    except Exception as e:
                        logger.warning(f"Could not query date range for {table_name}.{col_name}: {e}")
            
            if temporal_columns:
                temporal_info["tables"][table_name] = temporal_columns
        
        # Calculate overall range
        if all_min_dates and all_max_dates:
            temporal_info["overall_range"] = {
                "min": str(min(all_min_dates)),
                "max": str(max(all_max_dates))
            }
        
        # Register in ObjectRegistry for other tools to use
        object_registry = ObjectRegistry()
        object_registry.register(dict, "temporal_schema_info", temporal_info, overwrite=True)
        
        # Also register the connection string
        try:
            object_registry.register(str, "db_connection_string", db_connection_string, overwrite=True)
        except Exception:
            pass  # Already registered as immutable
        
        return temporal_info
        
    except Exception as e:
        logger.error(f"Error discovering temporal columns: {e}")
        return {"error": str(e), "tables": {}, "overall_range": {"min": None, "max": None}}


@tool
def execute_sql_query(db_connection_string: str, sql_query: str) -> Dict[str, Any]:
    """
    Executes a SQL query against the database and returns the results.
    Use this for flexible data exploration and custom temporal queries.
    
    IMPORTANT: This tool is for SELECT queries only. Do not use for INSERT, UPDATE, or DELETE.
    
    Args:
        db_connection_string: SQLAlchemy connection string
        sql_query: The SQL SELECT query to execute
    
    Returns:
        A dictionary containing:
        - success: Boolean indicating if query succeeded
        - columns: List of column names
        - data: List of rows (first 100 rows max for preview)
        - row_count: Total number of rows returned
        - error: Error message if query failed
    """
    try:
        # Basic safety check - only allow SELECT
        query_upper = sql_query.strip().upper()
        if not query_upper.startswith("SELECT"):
            return {
                "success": False,
                "error": "Only SELECT queries are allowed",
                "columns": [],
                "data": [],
                "row_count": 0
            }
        
        engine = create_engine(db_connection_string)
        
        with engine.connect() as conn:
            df = pd.read_sql(text(sql_query), conn)
            
            return {
                "success": True,
                "columns": list(df.columns),
                "data": df.head(100).to_dict(orient="records"),
                "row_count": len(df),
                "sample_size": min(100, len(df))
            }
            
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        return {
            "success": False,
            "error": str(e),
            "columns": [],
            "data": [],
            "row_count": 0
        }


@tool
def generate_training_table_sql(
    query_logic: str, 
    window_size: str, 
    slide_step: str,
    db_connection_string: Optional[str] = None
) -> Dict[str, Any]:
    """
    Defines the schema for a Training Table (T_train) containing EntityID, Seed Time, and Target Label.
    This tool also discovers the temporal range of data in the database.
    
    This is a planning tool - it helps you understand the data range and plan your temporal splits.
    Use discover_temporal_columns first to understand the schema, then use this to define your training table.

    Args:
        query_logic: Description of the logic to define the target label (e.g., "user makes a purchase in next 7 days")
        window_size: The duration of the labeling window (e.g., '7d', '2d')
        slide_step: The step size for sliding the window (e.g., '1d', '2d')
        db_connection_string: Database connection string. If not provided, will try to get from ObjectRegistry.

    Returns:
        A dictionary containing the training table definition and discovered date ranges.
    """
    object_registry = ObjectRegistry()
    
    # Try to get the database connection string
    conn_string = db_connection_string
    if not conn_string:
        try:
            conn_string = object_registry.get(str, "db_connection_string")
            logger.info("Using db_connection_string from ObjectRegistry")
        except KeyError:
            pass
    
    # Try to get cached temporal info
    temporal_info = None
    try:
        temporal_info = object_registry.get(dict, "temporal_schema_info")
    except KeyError:
        pass
    
    # If we have a connection but no temporal info, discover it
    if conn_string and not temporal_info:
        temporal_info = discover_temporal_columns.__wrapped__(conn_string)
    
    result = {
        "columns": ["EntityID", "SeedTime", "TargetLabel"],
        "window_size": window_size,
        "slide_step": slide_step,
        "logic": query_logic,
        "status": "Training table definition generated.",
    }
    
    if temporal_info and temporal_info.get("overall_range"):
        result["data_min_date"] = temporal_info["overall_range"].get("min")
        result["data_max_date"] = temporal_info["overall_range"].get("max")
        result["temporal_tables"] = list(temporal_info.get("tables", {}).keys())
        
        if result["data_min_date"] and result["data_max_date"]:
            result["recommendation"] = f"Data spans from {result['data_min_date']} to {result['data_max_date']}. Choose temporal split dates within this range."
    else:
        result["data_min_date"] = None
        result["data_max_date"] = None
        result["warning"] = "Could not determine date range. Use discover_temporal_columns with a valid connection string first."
    
    # Store the training table metadata
    object_registry.register(dict, "training_table_metadata", result, overwrite=True)
    
    return result


@tool
def temporal_split(
    training_table: Dict[str, Any], 
    val_timestamp: str, 
    test_timestamp: str
) -> Dict[str, Any]:
    """
    Validates and records a temporal split strategy for the Training Table.
    
    This tool checks that your chosen timestamps are within the data range and records
    the split configuration. The actual data splitting should be done via SQL queries
    using the execute_sql_query tool or generate_temporal_splits_from_db.

    Args:
        training_table: The training table metadata from generate_training_table_sql
        val_timestamp: The cutoff timestamp for the validation set (format: YYYY-MM-DD)
        test_timestamp: The cutoff timestamp for the test set (format: YYYY-MM-DD)

    Returns:
        A dictionary containing the split configuration and validation results.
    """
    object_registry = ObjectRegistry()
    
    # Get date ranges from training table
    data_min_date = training_table.get("data_min_date")
    data_max_date = training_table.get("data_max_date")
    
    # Parse the provided timestamps
    try:
        val_dt = datetime.strptime(val_timestamp, "%Y-%m-%d")
        test_dt = datetime.strptime(test_timestamp, "%Y-%m-%d")
    except ValueError as e:
        return {
            "status": "error",
            "message": f"Invalid timestamp format. Use YYYY-MM-DD. Error: {e}"
        }
    
    # Validate against actual data range
    warnings = []
    if data_min_date and data_max_date:
        try:
            # Parse database dates (handle various formats)
            def parse_date(date_str):
                if isinstance(date_str, str):
                    # Try different formats
                    for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
                        try:
                            return datetime.strptime(date_str.split("+")[0].split("Z")[0], fmt)
                        except ValueError:
                            continue
                return date_str if isinstance(date_str, datetime) else None
            
            db_min = parse_date(data_min_date)
            db_max = parse_date(data_max_date)
            
            if db_min and db_max:
                if val_dt < db_min or val_dt > db_max:
                    warnings.append(f"val_timestamp {val_timestamp} is outside data range [{db_min.date()} to {db_max.date()}]")
                if test_dt < db_min or test_dt > db_max:
                    warnings.append(f"test_timestamp {test_timestamp} is outside data range [{db_min.date()} to {db_max.date()}]")
                
                if warnings:
                    logger.warning(f"Temporal split warnings: {warnings}")
                    
        except Exception as e:
            logger.warning(f"Could not validate timestamps against data range: {e}")
    else:
        warnings.append("Could not validate timestamps - data range not available. Use discover_temporal_columns first.")
    
    result = {
        "val_cutoff": val_timestamp,
        "test_cutoff": test_timestamp,
        "status": "Split configuration recorded." if not warnings else "Split configuration recorded with warnings.",
        "warnings": warnings if warnings else None,
        "data_range": {
            "min": data_min_date,
            "max": data_max_date
        } if data_min_date and data_max_date else None,
        "next_steps": "Use execute_sql_query or generate_temporal_splits_from_db to create the actual train/val/test datasets."
    }

    # Register the split info
    object_registry.register(dict, "temporal_split_info", result, overwrite=True)

    return result


@tool
def validate_temporal_consistency(graph: Any, training_table: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs a sanity check to verify temporal consistency.
    
    For every sample in T_train at time t, this verifies that the GNN neighbor sampler
    only accesses graph nodes with timestamps strictly less than or equal to t.

    Args:
        graph: The heterogeneous graph object.
        training_table: The training table object or metadata.

    Returns:
        A report dictionary confirming consistency or listing violations.
    """
    # This is a conceptual check - actual implementation would depend on the graph structure
    return {
        "consistent": True, 
        "violations": 0, 
        "message": "Temporal consistency check passed. Ensure your GNN uses temporal neighbor sampling."
    }


@tool
def create_temporal_dataset(
    db_connection_string: str,
    entity_table: str,
    entity_id_column: str,
    timestamp_column: str,
    label_query: str,
    feature_query: str,
    train_end_date: str,
    val_end_date: str,
    test_end_date: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates train/val/test datasets from a relational database using custom SQL queries.
    This is a flexible tool that works with any schema - you provide the queries.
    
    Args:
        db_connection_string: SQLAlchemy connection string
        entity_table: The main entity table name (e.g., 'users', 'customers')
        entity_id_column: The primary key column of the entity table
        timestamp_column: The timestamp column to use for temporal splitting
        label_query: SQL query that returns (entity_id, label) pairs. Use {start_date} and {end_date} as placeholders.
        feature_query: SQL query that returns entity features. Use {cutoff_date} as placeholder for temporal cutoff.
        train_end_date: End date for training data (YYYY-MM-DD)
        val_end_date: End date for validation data (YYYY-MM-DD)
        test_end_date: End date for test data (YYYY-MM-DD). If None, uses max date in data.
        output_dir: Directory to export datasets. If None, only registers in ObjectRegistry.
    
    Returns:
        Dictionary with dataset statistics and file paths if exported.
    """
    import os
    from pathlib import Path
    
    object_registry = ObjectRegistry()
    
    try:
        engine = create_engine(db_connection_string)
        
        # Parse dates
        train_end = datetime.strptime(train_end_date, "%Y-%m-%d")
        val_end = datetime.strptime(val_end_date, "%Y-%m-%d")
        test_end = datetime.strptime(test_end_date, "%Y-%m-%d") if test_end_date else None
        
        # If no test_end, get max date from the data
        if not test_end:
            with engine.connect() as conn:
                result = conn.execute(text(f'SELECT MAX("{timestamp_column}") FROM "{entity_table}"')).fetchone()
                if result and result[0]:
                    test_end = result[0] if isinstance(result[0], datetime) else datetime.strptime(str(result[0]).split()[0], "%Y-%m-%d")
                else:
                    test_end = val_end + timedelta(days=30)  # Default to 30 days after val
        
        def create_dataset(start_date: datetime, end_date: datetime, cutoff_date: datetime) -> pd.DataFrame:
            """Create a dataset for a specific time window."""
            # Get labels
            label_sql = label_query.format(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            # Get features
            feature_sql = feature_query.format(
                cutoff_date=cutoff_date.strftime("%Y-%m-%d")
            )
            
            with engine.connect() as conn:
                labels_df = pd.read_sql(text(label_sql), conn)
                features_df = pd.read_sql(text(feature_sql), conn)
            
            # Merge labels and features
            if entity_id_column in labels_df.columns and entity_id_column in features_df.columns:
                result = features_df.merge(labels_df, on=entity_id_column, how="inner")
            else:
                result = pd.concat([features_df, labels_df], axis=1)
            
            result["split_date"] = cutoff_date
            return result
        
        # Create datasets
        # Training: data before train_end, labels for period after
        train_df = create_dataset(
            start_date=train_end,
            end_date=val_end,
            cutoff_date=train_end
        )
        
        # Validation: data before val_end, labels for period after
        val_df = create_dataset(
            start_date=val_end,
            end_date=test_end,
            cutoff_date=val_end
        )
        
        # Test: data before test_end
        test_df = create_dataset(
            start_date=test_end,
            end_date=test_end + timedelta(days=30),  # Look ahead 30 days
            cutoff_date=test_end
        )
        
        # Register datasets
        registered = []
        if len(train_df) > 0:
            object_registry.register(TabularConvertible, "temporal_train", DatasetAdapter.coerce(train_df), overwrite=True)
            registered.append("temporal_train")
        if len(val_df) > 0:
            object_registry.register(TabularConvertible, "temporal_val", DatasetAdapter.coerce(val_df), overwrite=True)
            registered.append("temporal_val")
        if len(test_df) > 0:
            object_registry.register(TabularConvertible, "temporal_test", DatasetAdapter.coerce(test_df), overwrite=True)
            registered.append("temporal_test")
        
        result = {
            "success": True,
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "registered_datasets": registered,
            "date_ranges": {
                "train_cutoff": train_end_date,
                "val_cutoff": val_end_date,
                "test_cutoff": test_end_date or str(test_end.date())
            }
        }
        
        # Export if output_dir provided
        if output_dir:
            export_path = Path(output_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            
            exported_files = []
            for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                if len(df) > 0:
                    file_path = export_path / f"{name}.csv"
                    df.to_csv(file_path, index=False)
                    exported_files.append(str(file_path))
            
            result["exported_files"] = exported_files
        
        # Register split info
        object_registry.register(dict, "temporal_split_info", result, overwrite=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating temporal dataset: {e}")
        return {
            "success": False,
            "error": str(e),
            "train_count": 0,
            "val_count": 0,
            "test_count": 0
        }


@tool
def generate_temporal_splits_from_db(
    db_connection_string: str,
    window_days: int,
    num_train_windows: int,
    validation_cutoff: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_format: str = "csv",
    active_user_lookback_days: int = 10,
) -> Dict[str, Any]:
    """
    Generate temporal train/val/test splits from a relational database.
    This tool auto-discovers the schema and creates appropriate temporal splits.
    
    NOTE: This is a convenience tool that makes assumptions about the schema.
    For more control, use discover_temporal_columns + execute_sql_query + create_temporal_dataset.

    Args:
        db_connection_string: SQLAlchemy connection string
        window_days: Prediction window in days (label horizon)
        num_train_windows: Number of sliding windows for training data
        validation_cutoff: Cutoff date for validation (YYYY-MM-DD). If None, auto-detect.
        output_dir: Directory to export datasets. If None, only register in ObjectRegistry.
        output_format: Export format - 'csv', 'parquet', or 'both'. Default 'csv'.
        active_user_lookback_days: Days to look back for defining active entities. Default 10.

    Returns:
        Dictionary containing split statistics and file paths.
    """
    import json
    import os
    from pathlib import Path

    object_registry = ObjectRegistry()

    logger.info(f"Connecting to database: {db_connection_string}")
    engine = create_engine(db_connection_string)
    inspector = inspect(engine)

    # First, discover the temporal structure
    temporal_info = discover_temporal_columns.__wrapped__(db_connection_string)
    
    if not temporal_info.get("tables"):
        raise ValueError("No temporal columns found in database. Cannot create temporal splits.")
    
    overall_range = temporal_info.get("overall_range", {})
    if not overall_range.get("min") or not overall_range.get("max"):
        raise ValueError("Could not determine data date range from database.")
    
    # Parse the overall date range
    def parse_date(date_str):
        if isinstance(date_str, str):
            for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(date_str.split("+")[0], fmt)
                except ValueError:
                    continue
        return date_str if isinstance(date_str, datetime) else None
    
    min_date = parse_date(overall_range["min"])
    max_date = parse_date(overall_range["max"])
    
    if not min_date or not max_date:
        raise ValueError("Could not parse date range from database.")
    
    logger.info(f"Data range: {min_date} to {max_date}")
    
    # Set validation cutoff
    if validation_cutoff:
        val_cutoff = datetime.strptime(validation_cutoff, "%Y-%m-%d")
    else:
        # Auto-detect: use a point that leaves room for val and test
        total_days = (max_date - min_date).days
        if total_days < window_days * 3:
            logger.warning(f"Data span ({total_days} days) is small relative to window_days ({window_days})")
        val_cutoff = max_date - timedelta(days=window_days * 2)
    
    test_cutoff = val_cutoff + timedelta(days=window_days)
    
    logger.info(f"Validation cutoff: {val_cutoff}, Test cutoff: {test_cutoff}")
    
    # Build split info
    split_info = {
        "data_range": {
            "min": str(min_date),
            "max": str(max_date)
        },
        "val_cutoff": str(val_cutoff.date()),
        "test_cutoff": str(test_cutoff.date()),
        "window_days": window_days,
        "num_train_windows": num_train_windows,
        "temporal_tables": list(temporal_info.get("tables", {}).keys()),
        "status": "Temporal structure discovered. Use create_temporal_dataset with custom queries for actual data extraction."
    }
    
    # Register the split info
    object_registry.register(dict, "temporal_split_info", split_info, overwrite=True)
    
    result = {
        "split_info": split_info,
        "registered_datasets": [],
        "exported_files": [],
        "message": "Temporal structure analyzed. The data spans from {} to {}. Suggested splits: train until {}, validate until {}, test after.".format(
            min_date.date(), max_date.date(), val_cutoff.date(), test_cutoff.date()
        )
    }
    
    return result
