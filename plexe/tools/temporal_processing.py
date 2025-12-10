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
from plexe.config import config

logger = logging.getLogger(__name__)


# =============================================================================
# Unified Schema Cache Key - Shared with graph_processing.py
# =============================================================================
# Both TemporalSupervisor and GraphArchitect need schema information.
# We use a unified cache key to avoid redundant DB queries.

UNIFIED_SCHEMA_CACHE_KEY = "unified_db_schema"


def _get_cached_temporal_info(db_connection: str) -> Optional[Dict[str, Any]]:
    """
    Check if temporal info is already cached in ObjectRegistry.
    Returns cached temporal info if connection string matches, None otherwise.
    """
    try:
        object_registry = ObjectRegistry()
        cached = object_registry.get(dict, UNIFIED_SCHEMA_CACHE_KEY)
        
        # Verify connection string matches
        if cached.get("_connection_string") == db_connection:
            # Extract temporal info from unified schema
            if "tables" in cached:
                logger.info("Extracting temporal info from cached unified schema")
                temporal_info = {"tables": {}, "overall_range": cached.get("temporal_summary", {}).get("overall_date_range", {"min": None, "max": None})}
                
                for table_name, table_data in cached.get("tables", {}).items():
                    temporal_ranges = table_data.get("temporal_ranges")
                    if temporal_ranges:
                        temporal_columns = []
                        for col_name, ranges in temporal_ranges.items():
                            temporal_columns.append({
                                "column": col_name,
                                "type": "TIMESTAMP",  # Approximate
                                "min_date": ranges.get("min"),
                                "max_date": ranges.get("max")
                            })
                        if temporal_columns:
                            temporal_info["tables"][table_name] = temporal_columns
                
                return temporal_info
    except KeyError:
        pass
    
    # Also check for dedicated temporal_schema_info
    try:
        object_registry = ObjectRegistry()
        temporal_info = object_registry.get(dict, "temporal_schema_info")
        if temporal_info:
            return temporal_info
    except KeyError:
        pass
        
    return None


def _cache_temporal_to_unified(db_connection: str, temporal_info: Dict[str, Any]) -> None:
    """Cache temporal info and update unified schema if it exists."""
    object_registry = ObjectRegistry()
    
    # Register dedicated temporal info
    object_registry.register(dict, "temporal_schema_info", temporal_info, overwrite=True)
    object_registry.register(str, "db_connection_string", db_connection, overwrite=True)
    
    # If unified schema exists, update it with temporal info
    try:
        cached = object_registry.get(dict, UNIFIED_SCHEMA_CACHE_KEY)
        if cached.get("_connection_string") == db_connection:
            # Update temporal summary in unified schema
            cached["temporal_summary"] = {
                "tables_with_timestamps": list(temporal_info.get("tables", {}).keys()),
                "overall_date_range": temporal_info.get("overall_range", {"min": None, "max": None})
            }
            object_registry.register(dict, UNIFIED_SCHEMA_CACHE_KEY, cached, overwrite=True)
            logger.info("Updated unified schema with temporal info")
    except KeyError:
        # No unified schema yet - create a minimal one for temporal info
        unified = {
            "_connection_string": db_connection,
            "temporal_summary": {
                "tables_with_timestamps": list(temporal_info.get("tables", {}).keys()),
                "overall_date_range": temporal_info.get("overall_range", {"min": None, "max": None})
            },
            "tables": {},  # Will be filled by extract_schema_metadata later
            "relationships": []
        }
        # Add basic temporal column info to tables
        for table_name, columns in temporal_info.get("tables", {}).items():
            unified["tables"][table_name] = {
                "temporal_columns": [c["column"] for c in columns],
                "temporal_ranges": {
                    c["column"]: {"min": c.get("min_date"), "max": c.get("max_date")}
                    for c in columns
                }
            }
        object_registry.register(dict, UNIFIED_SCHEMA_CACHE_KEY, unified, overwrite=True)
        logger.info("Created unified schema cache with temporal info")


def _discover_temporal_columns_impl(db_connection_string: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Internal implementation of discover_temporal_columns.
    """
    # Check cache first
    if not force_refresh:
        cached = _get_cached_temporal_info(db_connection_string)
        if cached and cached.get("tables"):
            logger.info("Returning cached temporal schema info")
            return cached
    
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
        
        # Cache in unified schema registry
        _cache_temporal_to_unified(db_connection_string, temporal_info)
        
        return temporal_info
        
    except Exception as e:
        logger.error(f"Error discovering temporal columns: {e}")
        return {"error": str(e), "tables": {}, "overall_range": {"min": None, "max": None}}


@tool
def discover_temporal_columns(db_connection_string: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Discovers all timestamp/date columns across all tables in the database.
    Use this to understand the temporal structure of the data before creating temporal splits.
    
    **IMPORTANT**: This tool uses a unified schema cache shared with GraphArchitect's
    `extract_schema_metadata`. If schema has already been analyzed, cached info is returned.
    
    Args:
        db_connection_string: SQLAlchemy connection string (e.g., 'postgresql+psycopg2://user:pass@host:port/db')
        force_refresh: If True, bypass cache and re-query database. Default False.
    
    Returns:
        A dictionary containing:
        - tables: Dict mapping table names to lists of temporal columns with their date ranges
        - overall_range: The min and max dates across all temporal columns
    """
    return _discover_temporal_columns_impl(db_connection_string, force_refresh)

@tool
def get_table_columns(db_connection_string: str, table_name: str) -> Dict[str, Any]:
    """
    Get the exact column names and data types for a specific table.
    
    **CRITICAL**: Use this tool to verify column names BEFORE writing SQL queries.
    PostgreSQL databases typically use snake_case column names (e.g., owner_user_id, creation_date).
    NEVER assume column names - always check with this tool first!
    
    Args:
        db_connection_string: SQLAlchemy connection string
        table_name: The name of the table to inspect
    
    Returns:
        A dictionary containing:
        - success: Boolean indicating if query succeeded
        - table_name: The table inspected
        - columns: List of column info dicts with 'name', 'type', 'nullable' keys
        - column_names: Simple list of column names for quick reference
        - error: Error message if query failed
        
    Example:
        get_table_columns(conn_str, "posts") might return:
        {
            "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "owner_user_id", "type": "INTEGER", "nullable": True},  # Note: snake_case!
                {"name": "creation_date", "type": "TIMESTAMP", "nullable": True}
            ],
            "column_names": ["id", "owner_user_id", "creation_date", ...]
        }
    """
    try:
        engine = create_engine(db_connection_string)
        inspector = inspect(engine)
        
        # Check if table exists
        table_names = inspector.get_table_names()
        if table_name not in table_names:
            return {
                "success": False,
                "table_name": table_name,
                "error": f"Table '{table_name}' not found. Available tables: {table_names}",
                "columns": [],
                "column_names": []
            }
        
        columns = inspector.get_columns(table_name)
        column_info = []
        column_names = []
        
        for col in columns:
            col_info = {
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True)
            }
            column_info.append(col_info)
            column_names.append(col["name"])
        
        result = {
            "success": True,
            "table_name": table_name,
            "columns": column_info,
            "column_names": column_names,
            "hint": "Use these exact column names (typically snake_case) in your SQL queries."
        }
        
        # Register column info in ObjectRegistry for use by other tools
        object_registry = ObjectRegistry()
        object_registry.register(dict, f"table_columns_{table_name}", result, overwrite=True)
        logger.info(f"Registered column info for table '{table_name}': {column_names}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting table columns: {e}")
        return {
            "success": False,
            "table_name": table_name,
            "error": str(e),
            "columns": [],
            "column_names": []
        }


@tool
def validate_temporal_consistency(
    graph_obj_name: str, 
    training_table_name: str, 
    n_samples: int = 100,
    time_attr: str = 'time'
) -> Dict[str, Any]:
    """
    Performs a sampling-based sanity check to verify temporal consistency between the Graph and Training Table.
    
    This tool simulates the neighbor sampling process. It picks random entities from the training table 
    and checks their immediate neighbors in the graph. It reports the "Leakage Risk" - the percentage 
    of neighbors that strictly require temporal filtering because they are from the future.
    
    CRITICAL PRINCIPLE: A node at time t is only allowed to receive messages from neighbors w 
    where τ(w) <= t (neighbor timestamp must be <= node's seed time).

    Args:
        graph_obj_name: Name of the PyG HeteroData object in ObjectRegistry.
        training_table_name: Name of the training table (DataFrame) in ObjectRegistry.
        n_samples: Number of random training examples to check (default: 100).
        time_attr: The name of the timestamp attribute in the graph nodes (default: 'time').

    Returns:
        A report dictionary listing leakage risks and validation status.
    """
    import random
    import torch
    
    object_registry = ObjectRegistry()
    
    # 1. Retrieve Data
    try:
        # Graph is a PyG HeteroData object
        graph = object_registry.get(Any, graph_obj_name) 
        # Training table is a pandas DataFrame or TabularConvertible
        train_data = object_registry.get(Any, training_table_name)
        if hasattr(train_data, 'to_pandas'):
            df_train = train_data.to_pandas()
        elif isinstance(train_data, pd.DataFrame):
            df_train = train_data
        else:
            return {"status": "error", "message": f"Training table must be a DataFrame or TabularConvertible, got {type(train_data)}"}
    except KeyError as e:
        return {"status": "error", "message": f"Object not found in registry: {e}"}

    # 2. Validate Graph Structure
    if not hasattr(graph, 'node_types') or not hasattr(graph, 'edge_types'):
        return {
            "status": "error", 
            "message": "Graph object does not appear to be a PyG HeteroData. Missing node_types or edge_types."
        }

    # 3. Validate Timestamp Existence on Graph Nodes
    # Per RDL paper (Appendix A): Time mapping function τ must exist
    missing_time_nodes = []
    for node_type in graph.node_types:
        if time_attr not in graph[node_type]:
            missing_time_nodes.append(node_type)
    
    if missing_time_nodes:
        return {
            "status": "failed",
            "message": f"Graph nodes {missing_time_nodes} are missing the '{time_attr}' attribute. "
                       f"Temporal sampling requires timestamps on nodes. "
                       f"Use 'Temporal' modality in encode_multi_modal_features to add timestamps."
        }

    # 4. Monte Carlo Simulation Check
    violations = 0
    total_neighbors_checked = 0
    future_neighbors_detected = 0
    checked_samples = 0
    violation_details = []
    
    # Sample n random examples from training table
    sample_size = min(n_samples, len(df_train))
    samples = df_train.sample(sample_size)
    
    # Detect column names for entity_id and timestamp
    # Common column name patterns
    entity_col = None
    timestamp_col = None
    entity_type_col = None
    
    for col in df_train.columns:
        col_lower = col.lower()
        if entity_col is None and any(x in col_lower for x in ['entity_id', 'entityid', 'user_id', 'userid', 'id']):
            entity_col = col
        if timestamp_col is None and any(x in col_lower for x in ['timestamp', 'time', 'seed_time', 'seedtime', 'date']):
            timestamp_col = col
        if entity_type_col is None and any(x in col_lower for x in ['entity_type', 'entitytype', 'node_type', 'nodetype']):
            entity_type_col = col
    
    if entity_col is None or timestamp_col is None:
        return {
            "status": "error",
            "message": f"Could not detect entity_id or timestamp columns in training table. "
                       f"Available columns: {list(df_train.columns)}. "
                       f"Expected columns containing 'entity_id'/'id' and 'timestamp'/'time'."
        }
    
    logger.info(f"Detected columns - entity: {entity_col}, timestamp: {timestamp_col}, type: {entity_type_col}")
    
    for _, row in samples.iterrows():
        try:
            # Get seed time from Training Table (Section 2.2 of RDL paper)
            seed_time = row[timestamp_col]
            if isinstance(seed_time, str):
                seed_time = datetime.strptime(seed_time, "%Y-%m-%d").timestamp()
            elif isinstance(seed_time, datetime):
                seed_time = seed_time.timestamp()
            elif isinstance(seed_time, pd.Timestamp):
                seed_time = seed_time.timestamp()
            
            entity_id = int(row[entity_col])
            
            # Determine target node type
            if entity_type_col and entity_type_col in row:
                target_node_type = row[entity_type_col]
            else:
                # Default to first node type if not specified
                target_node_type = graph.node_types[0]
            
            # Validate entity exists in graph
            if target_node_type not in graph.node_types:
                logger.warning(f"Node type '{target_node_type}' not in graph")
                continue
                
            num_nodes = graph[target_node_type].num_nodes
            if entity_id >= num_nodes:
                logger.warning(f"Entity ID {entity_id} out of bounds for {target_node_type} (max: {num_nodes-1})")
                continue

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error processing row: {e}")
            continue

        # Validate: Node time <= Seed Time (Node must exist at training time)
        node_time_tensor = graph[target_node_type][time_attr][entity_id]
        node_time = node_time_tensor.item() if torch.is_tensor(node_time_tensor) else float(node_time_tensor)
        
        if node_time > seed_time:
            violations += 1
            violation_details.append({
                "entity_id": entity_id,
                "node_type": target_node_type,
                "node_time": node_time,
                "seed_time": seed_time,
                "issue": "entity_not_yet_exists"
            })
            continue

        # 5. Check 1-hop Neighbors for Future Leakage
        # Per RDL paper: Message passing must respect τ(neighbor) <= t
        for edge_type in graph.edge_types:
            src_type, rel, dst_type = edge_type
            
            # Check edges where current node is the destination (receiving messages)
            if dst_type == target_node_type:
                edge_index = graph[edge_type].edge_index
                
                # Find edges pointing to this entity
                mask = edge_index[1] == entity_id
                neighbor_indices = edge_index[0][mask]
                
                if len(neighbor_indices) == 0:
                    continue
                
                # Get timestamps of neighbors
                neighbor_times_tensor = graph[src_type][time_attr][neighbor_indices]
                if torch.is_tensor(neighbor_times_tensor):
                    neighbor_times = neighbor_times_tensor
                else:
                    neighbor_times = torch.tensor(neighbor_times_tensor)
                
                # CRITICAL CHECK: τ(neighbor) <= seed_time
                # Any neighbor with time > seed_time would cause data leakage
                future_mask = neighbor_times > seed_time
                n_future = future_mask.sum().item()
                
                total_neighbors_checked += len(neighbor_indices)
                future_neighbors_detected += n_future
                
                if n_future > 0:
                    # Record sample violation details (limit to first few)
                    if len(violation_details) < 10:
                        future_indices = neighbor_indices[future_mask][:3]  # First 3 examples
                        for idx in future_indices:
                            violation_details.append({
                                "entity_id": entity_id,
                                "node_type": target_node_type,
                                "seed_time": seed_time,
                                "neighbor_idx": idx.item(),
                                "neighbor_type": src_type,
                                "neighbor_time": graph[src_type][time_attr][idx].item(),
                                "edge_type": f"{src_type}-{rel}-{dst_type}",
                                "issue": "future_neighbor"
                            })

        checked_samples += 1

    # 6. Compile Report
    leakage_risk_ratio = (future_neighbors_detected / total_neighbors_checked) if total_neighbors_checked > 0 else 0
    
    # Determine overall status
    if violations > 0:
        status = "failed"
    elif future_neighbors_detected > 0:
        status = "warning"
    else:
        status = "passed"
    
    result = {
        "status": status,
        "checked_samples": checked_samples,
        "total_samples_requested": n_samples,
        "entity_timestamp_violations": violations,
        "total_neighbors_checked": total_neighbors_checked,
        "future_neighbors_detected": future_neighbors_detected,
        "leakage_risk_ratio": f"{leakage_risk_ratio:.2%}",
        "leakage_risk_ratio_numeric": leakage_risk_ratio,
        "temporal_consistency_verified": violations == 0,
        "requires_temporal_sampling": future_neighbors_detected > 0,
        "message": "",
        "violation_samples": violation_details[:10] if violation_details else None,
    }

    if violations > 0:
        result["message"] = (
            f"CRITICAL FAILURE: Found {violations} training examples where the entity itself "
            f"does not exist yet at seed_time. This indicates a fundamental error in training table construction. "
            f"Entities must exist (have timestamp <= seed_time) before they can be training targets."
        )
    elif future_neighbors_detected > 0:
        result["message"] = (
            f"WARNING: Detected {future_neighbors_detected}/{total_neighbors_checked} neighbors "
            f"({leakage_risk_ratio:.1%}) from the future relative to training seed times. "
            f"This is EXPECTED in the raw graph, but CONFIRMS that you MUST use a 'TemporalNeighborSampler' "
            f"or temporal filtering during GNN training to exclude these neighbors. "
            f"Without temporal sampling, DATA LEAKAGE WILL OCCUR."
        )
    else:
        result["message"] = (
            f"PERFECT: No future leakage detected in {checked_samples} sampled training examples. "
            f"Either the graph is static, already temporally pruned, or all neighbors respect τ(w) <= t."
        )

    # Register the validation result
    object_registry.register(dict, "temporal_consistency_report", result, overwrite=True)
    
    logger.info(f"Temporal consistency check: {status} - {result['message'][:100]}...")

    return result


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
    window_size_days: int = 0,
    num_train_windows: int = 1,
    train_stride_days: int = 30,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates train/val/test datasets using Relational Deep Learning principles.
    
    This tool implements the SLIDING WINDOW SAMPLING strategy from the RDL paper (Figure 3b).
    Training data is generated by moving backwards in time from train_end_date, creating
    multiple snapshots to maximize training samples while maintaining temporal consistency.
    
    CRITICAL PRINCIPLES:
    1. Training samples use sliding windows moving back in time (multiple seed_times)
    2. Validation/Test use single snapshots at their respective cutoff dates
    3. Gap between val and test must be >= window_size_days to prevent label leakage
    4. Features are computed using data strictly BEFORE seed_time (point-in-time correctness)
    5. Labels are computed from [seed_time, seed_time + window_size_days]
    
    Args:
        db_connection_string: SQLAlchemy connection string
        entity_table: The main entity table name (e.g., 'users', 'customers')
        entity_id_column: The primary key column of the entity table
        timestamp_column: The timestamp column to use for temporal splitting
        label_query: SQL query returning (entity_id, label). Use {start_date} and {end_date} placeholders.
        feature_query: SQL query returning entity features. Use {cutoff_date} placeholder for temporal cutoff.
        train_end_date: The LATEST date for training samples (upper bound). Training windows move backwards from here.
        val_end_date: Cutoff date for validation set (single snapshot).
        test_end_date: Cutoff date for test set (single snapshot). If None, auto-detected.
        window_size_days: Prediction window size in days (e.g., 2 for "predict in next 2 days").
        num_train_windows: Number of historical snapshots for training (sliding window count).
        train_stride_days: Step size (days) to move back between training windows.
        output_dir: Directory to export datasets as CSV. If None, only registers in ObjectRegistry.
    
    Returns:
        Dictionary with dataset statistics, warnings, and file paths if exported.
    
    Example:
        If train_end_date="2012-01-10", num_train_windows=5, train_stride_days=30:
        - Training snapshots at: 2012-01-10, 2011-12-11, 2011-11-11, 2011-10-12, 2011-09-12
        - Each snapshot: features <= seed_time, labels in [seed_time, seed_time + window_size_days]
    """
    import os
    from pathlib import Path
    
    object_registry = ObjectRegistry()
    warnings = []

    # Default to cache dir if not provided to ensure persistence across agents
    if output_dir is None:
        output_dir = config.file_storage.cache_dir
    
    try:
        engine = create_engine(db_connection_string)
        
        # Parse dates
        train_end_cutoff = datetime.strptime(train_end_date, "%Y-%m-%d")
        val_cutoff = datetime.strptime(val_end_date, "%Y-%m-%d")
        test_cutoff = datetime.strptime(test_end_date, "%Y-%m-%d") if test_end_date else None
        
        # Window delta for label computation
        window_delta = timedelta(days=window_size_days) if window_size_days > 0 else timedelta(days=0)
        
        # Auto-detect test cutoff if not provided
        if not test_cutoff:
            with engine.connect() as conn:
                result = conn.execute(text(f'SELECT MAX("{timestamp_column}") FROM "{entity_table}"')).fetchone()
                if result and result[0]:
                    max_date = result[0] if isinstance(result[0], datetime) else datetime.strptime(str(result[0]).split()[0], "%Y-%m-%d")
                    # Set test cutoff to ensure room for labels
                    test_cutoff = max_date - window_delta
                else:
                    test_cutoff = val_cutoff + timedelta(days=max(window_size_days, 30))
            logger.info(f"Auto-detected test_cutoff: {test_cutoff.date()}")
        
        # 1. CRITICAL LEAKAGE CHECKS
        # Check 1: Val cutoff must be before test cutoff
        if val_cutoff >= test_cutoff:
            error_msg = f"INVALID: val_end_date ({val_end_date}) must be before test_end_date ({test_cutoff.date()})"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Check 2: Training must end before validation starts
        if train_end_cutoff > val_cutoff:
            warn_msg = (
                f"WARNING: train_end_date ({train_end_date}) is after val_end_date ({val_end_date}). "
                f"Training samples should be strictly historical relative to validation."
            )
            warnings.append(warn_msg)
            logger.warning(warn_msg)
        
        # Check 3: Gap between val and test must accommodate prediction window
        gap_days = (test_cutoff - val_cutoff).days
        if window_size_days > 0 and gap_days < window_size_days:
            leakage_msg = (
                f"DATA LEAKAGE RISK: Gap between val and test is {gap_days} days, "
                f"but prediction window is {window_size_days} days. "
                f"Validation labels (computed until {(val_cutoff + window_delta).date()}) "
                f"will overlap with test input period (starting {test_cutoff.date()})."
            )
            warnings.append(leakage_msg)
            logger.warning(leakage_msg)
        
        # Helper function to generate ONE snapshot at a specific seed_time
        def fetch_snapshot(seed_time: datetime, split_name: str) -> pd.DataFrame:
            """
            Generate a dataset snapshot at the given seed_time.
            
            Features: Computed from data with timestamp <= seed_time (point-in-time correctness)
            Labels: Computed from data in [seed_time, seed_time + window_delta]
            """
            # Label window: [seed_time, seed_time + window]
            label_start = seed_time
            label_end = seed_time + window_delta if window_size_days > 0 else seed_time + timedelta(days=1)
            
            # Build SQL queries with date placeholders
            label_sql = label_query.format(
                start_date=label_start.strftime("%Y-%m-%d"),
                end_date=label_end.strftime("%Y-%m-%d")
            )
            
            # Features use data strictly before seed_time
            feature_sql = feature_query.format(
                cutoff_date=seed_time.strftime("%Y-%m-%d")
            )
            
            try:
                with engine.connect() as conn:
                    labels_df = pd.read_sql(text(label_sql), conn)
                    features_df = pd.read_sql(text(feature_sql), conn)
            except Exception as e:
                logger.error(f"SQL error for {split_name} at {seed_time}: {e}")
                return pd.DataFrame()
            
            # Merge labels and features
            # Robustly find entity_id column in both dataframes (case-insensitive)
            feat_id_col = next((c for c in features_df.columns if c.lower() == entity_id_column.lower()), None)
            lbl_id_col = next((c for c in labels_df.columns if c.lower() == entity_id_column.lower()), None)
            
            # If not found, try common ID names to rescue the merge
            if not feat_id_col:
                feat_id_col = next((c for c in features_df.columns if c.lower() in ['id', 'user_id', 'item_id', 'entity_id', 'post_id']), None)
            if not lbl_id_col:
                lbl_id_col = next((c for c in labels_df.columns if c.lower() in ['id', 'user_id', 'item_id', 'entity_id', 'post_id']), None)

            if feat_id_col and lbl_id_col:
                df = features_df.merge(labels_df, left_on=feat_id_col, right_on=lbl_id_col, how="inner")
                # If names differ, keep the one matching entity_id_column or the first one
                final_id_col = feat_id_col
            else:
                # Fallback: concat (unsafe, may have misaligned rows)
                logger.warning(f"Entity column '{entity_id_column}' not found in both dataframes (feat={feat_id_col}, lbl={lbl_id_col}). Using concat.")
                df = pd.concat([features_df, labels_df], axis=1)
                # Remove duplicate columns to prevent AttributeError: 'DataFrame' object has no attribute 'dtype'
                df = df.loc[:, ~df.columns.duplicated()]
                final_id_col = feat_id_col if feat_id_col else (lbl_id_col if lbl_id_col else df.columns[0])
            
            # Add metadata columns (important for Temporal GNN training)
            df["seed_time"] = seed_time
            # df["seed_time_str"] = seed_time.strftime("%Y-%m-%d") # Removed to reduce columns
            # df["split"] = split_name # Removed to reduce columns
            
            # Filter columns to match RDL Training Table format: [EntityID, SeedTime, Label]
            # We assume the label column is named 'label' or is the last column in labels_df
            label_col = "label"
            if label_col not in df.columns:
                # Try to find it in labels_df
                label_candidates = [c for c in labels_df.columns if c.lower() != entity_id_column.lower()]
                if label_candidates:
                    label_col = label_candidates[0]
            
            # Select and reorder columns
            cols_to_keep = [final_id_col, "seed_time"]
            if label_col in df.columns:
                cols_to_keep.append(label_col)
            
            df = df[cols_to_keep]
            
            # Rename ID column to standard 'entity_id' if requested, or keep original
            # The user wants 3 columns. Let's standardize to entity_id_column name provided
            if final_id_col != entity_id_column:
                 df = df.rename(columns={final_id_col: entity_id_column})

            return df
        
        # 2. GENERATE TRAINING DATA (Sliding Window Strategy - RDL Paper Figure 3b)
        logger.info(f"Generating {num_train_windows} training windows, ending at {train_end_cutoff.date()}, stride={train_stride_days}d")
        
        train_dfs = []
        current_seed = train_end_cutoff
        train_window_dates = []
        
        for i in range(num_train_windows):
            # Ensure we don't go before validation cutoff with training labels
            # Training labels end at current_seed + window_delta
            # This should be strictly before val_cutoff to avoid leakage
            if window_size_days > 0 and current_seed + window_delta > val_cutoff:
                warn_msg = f"Skipping train window at {current_seed.date()}: labels would extend past val_cutoff"
                logger.warning(warn_msg)
                current_seed -= timedelta(days=train_stride_days)
                continue
            
            snapshot = fetch_snapshot(current_seed, "train")
            
            if not snapshot.empty:
                # Check for degenerate labels (all 0 or all null)
                label_col = next((c for c in snapshot.columns if c.lower() not in [entity_id_column.lower(), 'seed_time']), None)
                if label_col:
                    is_all_null = snapshot[label_col].isnull().all()
                    is_all_zero = (pd.to_numeric(snapshot[label_col], errors='coerce').fillna(0) == 0).all()
                    
                    if is_all_null:
                        logger.warning(f"  Window {i+1}: seed={current_seed.date()} - ALL LABELS ARE NULL. Check label query.")
                    elif is_all_zero:
                        logger.warning(f"  Window {i+1}: seed={current_seed.date()} - ALL LABELS ARE ZERO. Check label query.")
                    else:
                        train_dfs.append(snapshot)
                        train_window_dates.append(current_seed.strftime("%Y-%m-%d"))
                        logger.debug(f"  Window {i+1}: seed={current_seed.date()}, samples={len(snapshot)}")
                else:
                    # No label column found?
                    logger.warning(f"  Window {i+1}: seed={current_seed.date()} - NO LABEL COLUMN FOUND.")
            else:
                logger.warning(f"  Window {i+1}: seed={current_seed.date()}, NO DATA")
            
            # Move backwards in time for next window
            current_seed -= timedelta(days=train_stride_days)
        
        # Combine all training windows
        train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        logger.info(f"Training set: {len(train_df)} samples from {len(train_dfs)} windows")
        
        # Check if training set is empty or degenerate
        if train_df.empty:
            return {
                "success": False,
                "error": "Generated training set is empty. Check your queries and date ranges.",
                "stats": {"train_total_samples": 0}
            }
            
        # 3. GENERATE VALIDATION DATA (Single Snapshot)
        logger.info(f"Generating validation snapshot at {val_cutoff.date()}")
        val_df = fetch_snapshot(val_cutoff, "val")
        logger.info(f"Validation set: {len(val_df)} samples")
        
        # 4. GENERATE TEST DATA (Single Snapshot)
        logger.info(f"Generating test snapshot at {test_cutoff.date()}")
        test_df = fetch_snapshot(test_cutoff, "test")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # 5. REGISTER DATASETS in ObjectRegistry
        registered = []
        if not train_df.empty:
            object_registry.register(TabularConvertible, "temporal_train", DatasetAdapter.coerce(train_df), overwrite=True)
            registered.append("temporal_train")
        if not val_df.empty:
            object_registry.register(TabularConvertible, "temporal_val", DatasetAdapter.coerce(val_df), overwrite=True)
            registered.append("temporal_val")
        if not test_df.empty:
            object_registry.register(TabularConvertible, "temporal_test", DatasetAdapter.coerce(test_df), overwrite=True)
            registered.append("temporal_test")
        
        # 6. BUILD RESULT
        result = {
            "success": True,
            "stats": {
                "train_total_samples": len(train_df),
                "train_windows_generated": len(train_dfs),
                "train_windows_requested": num_train_windows,
                "train_stride_days": train_stride_days,
                "val_samples": len(val_df),
                "test_samples": len(test_df),
            },
            "registered_datasets": registered,
            "date_config": {
                "train_end_date": train_end_date,
                "train_window_dates": train_window_dates,
                "val_cutoff": val_end_date,
                "test_cutoff": test_end_date or str(test_cutoff.date()),
                "window_size_days": window_size_days,
                "gap_val_test_days": gap_days,
            },
            "split_summary": {
                "train": f"{len(train_dfs)} windows from {train_window_dates[-1] if train_window_dates else 'N/A'} to {train_window_dates[0] if train_window_dates else 'N/A'}, stride={train_stride_days}d",
                "val": f"snapshot at {val_end_date}, labels: [{val_end_date}, {(val_cutoff + window_delta).strftime('%Y-%m-%d')}]",
                "test": f"snapshot at {test_cutoff.date()}, labels: [{test_cutoff.date()}, {(test_cutoff + window_delta).strftime('%Y-%m-%d')}]",
            },
            "warnings": warnings if warnings else None,
        }
        
        # 7. EXPORT TO FILES if requested
        if output_dir:
            export_path = Path(output_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            
            exported_files = []
            for name, df in [("temporal_train", train_df), ("temporal_val", val_df), ("temporal_test", test_df)]:
                if not df.empty:
                    file_path = export_path / f"{name}.csv"
                    df.to_csv(file_path, index=False)
                    exported_files.append(str(file_path))
                    logger.info(f"Exported {name} to {file_path}")
            
            result["exported_files"] = exported_files
        
        # 8. REGISTER SPLIT INFO for other tools
        split_info = {
            "val_cutoff": val_end_date,
            "test_cutoff": test_end_date or str(test_cutoff.date()),
            "window_size_days": window_size_days,
            "gap_between_splits_days": gap_days,
            "num_train_windows": len(train_dfs),
            "train_stride_days": train_stride_days,
            "status": "Datasets created successfully." if not warnings else "Datasets created with warnings.",
            "warnings": warnings if warnings else None,
        }
        object_registry.register(dict, "temporal_split_info", split_info, overwrite=True)
        
        logger.info(f"Temporal dataset creation complete: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
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

