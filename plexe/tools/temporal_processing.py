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
def define_training_task(
    entity_table: str,
    entity_id_column: str,
    target_definition: str,
    prediction_window: str,
    slide_step: str,
    event_table: Optional[str] = None,
    event_timestamp_column: Optional[str] = None,
    entity_created_at_column: Optional[str] = None,
    prediction_horizon: Optional[str] = None,
    db_connection_string: Optional[str] = None
) -> Dict[str, Any]:
    """
    Step 1: Define the training task metadata and validate against DB time range.
    
    This tool defines the conceptual structure of a Training Table (T_train) following the RDL paper:
    - EntityID: The primary entity for prediction (e.g., user_id)
    - SeedTime: The observation timestamp for feature computation
    - TargetLabel: The label to predict (computed based on target_definition)
    
    Use discover_temporal_columns first to understand the schema, then use this to define the task.
    After defining the task, use generate_sql_implementation to create executable SQL.

    Args:
        entity_table: The primary entity table name (e.g., 'users', 'customers')
        entity_id_column: The column name for entity ID in the entity table (e.g., 'user_id', 'customer_id')
        target_definition: Natural language description of the target label logic 
                          (e.g., "user makes a purchase in the next 7 days", 
                           "customer churns within prediction window")
        prediction_window: The duration for computing labels (e.g., '7d', '14d', '30d')
                          This defines the time span for label computation from seed time.
        slide_step: The step size for sliding the seed time window (e.g., '1d', '7d')
                   Smaller steps = more training examples but higher correlation.
        event_table: The table containing events/transactions for label computation
                    (e.g., 'transactions', 'orders', 'events'). Required for generating executable SQL.
        event_timestamp_column: The timestamp column in the event table (e.g., 'created_at', 'event_time').
                               Required for temporal filtering in label computation.
        entity_created_at_column: Optional. The column indicating when the entity was created 
                                 (e.g., 'created_at', 'registered_at'). Used to filter out entities 
                                 that don't exist yet at each seed_time, preventing future leakage 
                                 and reducing computational waste from inactive entities.
        prediction_horizon: Optional. How far into the future to predict (e.g., '1d', '7d').
                           If not provided, defaults to same as prediction_window.
                           Used to ensure proper temporal gap to prevent leakage.
        db_connection_string: Database connection string. If not provided, will try to get from ObjectRegistry.

    Returns:
        A dictionary containing:
        - status: Task definition status
        - task_schema: The schema of the training table [EntityID, SeedTime, TargetLabel]
        - metadata: Task configuration including entity table, windows, and date ranges
        - next_step: Instructions for generating SQL implementation
    """
    object_registry = ObjectRegistry()
    
    # Parse window sizes to days for validation
    def parse_duration_to_days(duration_str: str) -> int:
        """Parse duration string like '7d', '2w', '1m' to days."""
        duration_str = duration_str.strip().lower()
        if duration_str.endswith('d'):
            return int(duration_str[:-1])
        elif duration_str.endswith('w'):
            return int(duration_str[:-1]) * 7
        elif duration_str.endswith('m'):
            return int(duration_str[:-1]) * 30
        else:
            # Try to parse as integer (assume days)
            try:
                return int(duration_str)
            except ValueError:
                return 7  # Default to 7 days
    
    prediction_window_days = parse_duration_to_days(prediction_window)
    slide_step_days = parse_duration_to_days(slide_step)
    horizon_days = parse_duration_to_days(prediction_horizon) if prediction_horizon else prediction_window_days
    
    # Try to get the database connection string
    conn_string = db_connection_string
    if not conn_string:
        try:
            conn_string = object_registry.get(str, "db_connection_string")
            logger.info("Using db_connection_string from ObjectRegistry")
        except KeyError:
            pass
    
    # Try to get cached temporal info or discover it
    temporal_info = None
    try:
        temporal_info = object_registry.get(dict, "temporal_schema_info")
    except KeyError:
        logger.debug("Cache not found temporal_info, will re-discover.")
    
    # If we have a connection but no temporal info, discover it
    if conn_string and not temporal_info:
        # Use the internal function to avoid decorator overhead
        temporal_info = discover_temporal_columns.__wrapped__(conn_string)
    
    # Extract date range information
    min_date = None
    max_date = None
    temporal_tables = []
    
    if temporal_info and temporal_info.get("overall_range"):
        min_date = temporal_info["overall_range"].get("min")
        max_date = temporal_info["overall_range"].get("max")
        temporal_tables = list(temporal_info.get("tables", {}).keys())
    
    # Build the task metadata
    task_metadata = {
        "entity_table": entity_table,
        "entity_id_column": entity_id_column,
        "target_definition": target_definition,
        "prediction_window": prediction_window,
        "prediction_window_days": prediction_window_days,
        "slide_step": slide_step,
        "slide_step_days": slide_step_days,
        "prediction_horizon": prediction_horizon or prediction_window,
        "prediction_horizon_days": horizon_days,
        "event_table": event_table,
        "event_timestamp_column": event_timestamp_column,
        "entity_created_at_column": entity_created_at_column,
        "available_date_range": {
            "min": min_date,
            "max": max_date
        },
        "temporal_tables": temporal_tables
    }
    
    # Build recommendations
    recommendations = []
    warnings = []
    
    if min_date and max_date:
        recommendations.append(
            f"Data spans from {min_date} to {max_date}. "
            f"Choose temporal split dates within this range."
        )
        
        # Validate entity table exists in temporal tables
        if entity_table not in temporal_tables and temporal_tables:
            warnings.append(
                f"Entity table '{entity_table}' not found in tables with temporal columns. "
                f"Available temporal tables: {temporal_tables}. "
                f"Ensure proper join logic in SQL implementation."
            )
    else:
        warnings.append(
            "Could not determine date range. Use discover_temporal_columns with a valid connection string first."
        )
    
    # Validate slide_step <= prediction_window
    if slide_step_days > prediction_window_days:
        warnings.append(
            f"slide_step ({slide_step}) is larger than prediction_window ({prediction_window}). "
            f"This may result in gaps between training examples."
        )
    
    # Warn if event_table or event_timestamp_column not provided
    if not event_table:
        warnings.append(
            "event_table not specified. You will need to provide it in generate_sql_implementation "
            "or replace the placeholder in the generated SQL."
        )
    if not event_timestamp_column:
        warnings.append(
            "event_timestamp_column not specified. You will need to provide it in generate_sql_implementation "
            "or replace the placeholder in the generated SQL."
        )
    
    # Recommend entity_created_at_column for efficiency
    if not entity_created_at_column:
        recommendations.append(
            "Consider providing entity_created_at_column (e.g., 'created_at', 'registered_at') "
            "to filter entities that exist at each seed_time. This prevents future leakage "
            "and significantly reduces computation on large entity tables."
        )
    
    result = {
        "status": "Task defined successfully",
        "task_schema": ["EntityID", "SeedTime", "TargetLabel"],
        "metadata": task_metadata,
        "recommendations": recommendations if recommendations else None,
        "warnings": warnings if warnings else None,
        "next_step": "Use generate_sql_implementation(task_metadata, dialect) to create executable SQL for this task."
    }
    
    # Store the task metadata for use by generate_sql_implementation
    object_registry.register(dict, "training_task_metadata", result, overwrite=True)
    
    # Also store with old key for backward compatibility
    object_registry.register(dict, "training_table_metadata", {
        "columns": ["EntityID", "SeedTime", "TargetLabel"],
        "window_size": prediction_window,
        "slide_step": slide_step,
        "logic": target_definition,
        "data_min_date": min_date,
        "data_max_date": max_date,
        "temporal_tables": temporal_tables,
        "status": "Training table definition generated."
    }, overwrite=True)
    
    logger.info(f"Training task defined: entity={entity_table}, window={prediction_window}, horizon={prediction_horizon or prediction_window}")
    
    return result


@tool
def generate_sql_implementation(
    task_metadata: Dict[str, Any],
    dialect: str = "postgresql",
    include_create_table: bool = False,
    custom_label_sql: Optional[str] = None,
    event_table: Optional[str] = None,
    event_timestamp_column: Optional[str] = None,
    entity_created_at_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Step 2: Generate executable SQL to create the training table based on task metadata.
    
    This tool transforms the task definition from define_training_task into actual SQL statements.
    It generates SQL for:
    1. Creating seed times with sliding window
    2. Filtering active entities at each seed time (if entity_created_at_column provided)
    3. Computing target labels based on the target_definition
    4. Joining with event table for label computation
    
    The generated SQL follows RDL paper principles:
    - Labels are computed from [SeedTime, SeedTime + prediction_window]
    - Features should only use data from (-∞, SeedTime] to prevent leakage
    - Only entities that exist at seed_time are included (active entity filtering)

    Args:
        task_metadata: The metadata dictionary from define_training_task (or the full result dict)
        dialect: SQL dialect to use ('postgresql', 'mysql', 'sqlite', 'bigquery'). Default: 'postgresql'
        include_create_table: If True, wraps the query in CREATE TABLE statement. Default: False
        custom_label_sql: Optional custom SQL expression for label computation. If not provided,
                         a template will be generated based on target_definition that needs LLM refinement.
        event_table: Override event table name. If not provided, uses value from task_metadata.
        event_timestamp_column: Override event timestamp column. If not provided, uses value from task_metadata.
        entity_created_at_column: Override entity created_at column. If not provided, uses value from task_metadata.
                                 When provided, filters entities to only include those created before seed_time,
                                 preventing future leakage and reducing computation on inactive entities.

    Returns:
        A dictionary containing:
        - sql_query: The generated SQL query (may need LLM refinement for complex label logic)
        - sql_create_table: CREATE TABLE version if include_create_table=True
        - parameters: Placeholders that need to be filled (if any)
        - notes: Important notes about the generated SQL
        - requires_refinement: Boolean indicating if LLM should refine the label logic
    """
    object_registry = ObjectRegistry()
    
    # Extract metadata - handle both direct metadata and wrapped result
    if "metadata" in task_metadata:
        meta = task_metadata["metadata"]
    else:
        meta = task_metadata
    
    entity_table = meta.get("entity_table", "entities")
    entity_id_column = meta.get("entity_id_column", "entity_id")
    target_definition = meta.get("target_definition", "")
    prediction_window = meta.get("prediction_window", "7d")
    prediction_window_days = meta.get("prediction_window_days", 7)
    slide_step = meta.get("slide_step", "1d")
    slide_step_days = meta.get("slide_step_days", 1)
    date_range = meta.get("available_date_range", {})
    min_date = date_range.get("min")
    max_date = date_range.get("max")
    
    # Get event table info - prioritize function params, then metadata
    evt_table = event_table or meta.get("event_table")
    evt_timestamp_col = event_timestamp_column or meta.get("event_timestamp_column")
    entity_created_col = entity_created_at_column or meta.get("entity_created_at_column")
    
    # Dialect-specific syntax
    dialect_config = {
        "postgresql": {
            "interval": lambda days: f"INTERVAL '{days} days'",
            "date_trunc": lambda col, unit: f"DATE_TRUNC('{unit}', {col})",
            "generate_series": True,
            "bool_type": "BOOLEAN",
            "timestamp_type": "TIMESTAMP"
        },
        "mysql": {
            "interval": lambda days: f"INTERVAL {days} DAY",
            "date_trunc": lambda col, unit: f"DATE({col})" if unit == "day" else f"DATE_FORMAT({col}, '%Y-%m-01')",
            "generate_series": False,
            "bool_type": "TINYINT(1)",
            "timestamp_type": "DATETIME"
        },
        "sqlite": {
            "interval": lambda days: f"'{days} days'",
            "date_trunc": lambda col, unit: f"DATE({col})",
            "generate_series": False,
            "bool_type": "INTEGER",
            "timestamp_type": "TEXT"
        },
        "bigquery": {
            "interval": lambda days: f"INTERVAL {days} DAY",
            "date_trunc": lambda col, unit: f"DATE_TRUNC({col}, {unit.upper()})",
            "generate_series": False,
            "bool_type": "BOOL",
            "timestamp_type": "TIMESTAMP"
        }
    }
    
    config = dialect_config.get(dialect.lower(), dialect_config["postgresql"])
    
    # Generate the base SQL structure
    notes = []
    parameters = {}
    requires_refinement = True
    
    # Placeholder for label logic - this is what needs LLM refinement
    if custom_label_sql:
        label_sql = custom_label_sql
        requires_refinement = False
        notes.append("Using custom label SQL provided by user.")
    else:
        # Generate a template based on target_definition
        label_sql = _generate_label_sql_template(target_definition, entity_id_column, config)
        notes.append(
            f"Label SQL is a TEMPLATE based on target_definition: '{target_definition}'. "
            "You should refine this SQL or provide custom_label_sql for accurate label computation."
        )
        parameters["label_logic"] = f"Refine based on: {target_definition}"
    
    # Generate seed time series SQL
    if config["generate_series"]:
        # PostgreSQL-style with generate_series
        seed_time_sql = f"""
-- Generate seed times using sliding window
WITH seed_times AS (
    SELECT generate_series(
        '{min_date}'::timestamp,
        '{max_date}'::timestamp - {config['interval'](prediction_window_days)},
        {config['interval'](slide_step_days)}
    ) AS seed_time
)"""
    else:
        # Fallback for dialects without generate_series
        seed_time_sql = f"""
-- Generate seed times using sliding window
-- NOTE: For {dialect}, you may need to create a calendar/numbers table
-- or use a recursive CTE to generate the seed time series
WITH RECURSIVE seed_times AS (
    SELECT CAST('{min_date}' AS {config['timestamp_type']}) AS seed_time
    UNION ALL
    SELECT seed_time + {config['interval'](slide_step_days)}
    FROM seed_times
    WHERE seed_time < CAST('{max_date}' AS {config['timestamp_type']}) - {config['interval'](prediction_window_days)}
)"""
    
    # Generate entity-seed pair logic based on whether we have entity_created_at_column
    # This addresses the "Active Entity Problem" - avoiding CROSS JOIN with all entities
    if entity_created_col:
        # OPTIMIZED: Only include entities that existed at the seed_time
        # This prevents future leakage and reduces computation significantly
        entity_seed_pairs_sql = f"""
-- Create pairs of (entity, seed_time) ONLY for entities that existed at that time
-- This prevents future leakage and reduces computation (Active Entity Filtering)
entity_seed_pairs AS (
    SELECT 
        e.{entity_id_column} AS entity_id,
        s.seed_time
    FROM {entity_table} e
    JOIN seed_times s ON e.{entity_created_col} <= s.seed_time
)"""
        notes.append(
            f"Active Entity Filtering enabled: Only entities with {entity_created_col} <= seed_time are included. "
            f"This prevents future leakage and reduces computation on inactive/future entities."
        )
    else:
        # FALLBACK: Cross join (less efficient, may include future entities)
        entity_seed_pairs_sql = f"""
-- Get all entities from the entity table
entities AS (
    SELECT DISTINCT {entity_id_column} AS entity_id
    FROM {entity_table}
),

-- Cross join to create all (entity, seed_time) combinations
-- WARNING: This may include entities that don't exist at the seed_time (future leakage risk)
-- Consider providing entity_created_at_column to filter active entities
entity_seed_pairs AS (
    SELECT 
        e.entity_id,
        s.seed_time
    FROM entities e
    CROSS JOIN seed_times s
)"""
        notes.append(
            "WARNING: Using CROSS JOIN for entity-seed pairs. This may include entities that don't exist "
            "at the seed_time (future leakage) and cause unnecessary computation on inactive entities. "
            "Provide entity_created_at_column to enable Active Entity Filtering."
        )
    
    # Determine event table reference in SQL
    if evt_table and evt_timestamp_col:
        # Fully specified - generate executable SQL
        event_join_sql = f"""
    LEFT JOIN {evt_table} evt ON 
        evt.{entity_id_column} = esp.entity_id
        AND evt.{evt_timestamp_col} >= esp.seed_time
        AND evt.{evt_timestamp_col} < esp.seed_time + {config['interval'](prediction_window_days)}"""
        notes.append(f"Event table: {evt_table}, timestamp column: {evt_timestamp_col}")
    else:
        # Need placeholders
        evt_table_ref = evt_table or "{{EVENT_TABLE}}"
        evt_ts_ref = evt_timestamp_col or "{{TIMESTAMP_COLUMN}}"
        event_join_sql = f"""
    -- TODO: Verify the event table and timestamp column names
    LEFT JOIN {evt_table_ref} evt ON 
        evt.{entity_id_column} = esp.entity_id
        AND evt.{evt_ts_ref} >= esp.seed_time
        AND evt.{evt_ts_ref} < esp.seed_time + {config['interval'](prediction_window_days)}"""
        
        if not evt_table:
            parameters["EVENT_TABLE"] = "The table containing events/transactions for label computation"
        if not evt_timestamp_col:
            parameters["TIMESTAMP_COLUMN"] = "The timestamp column in the event table"
    
    # Main query structure
    main_query = f"""
{seed_time_sql},

{entity_seed_pairs_sql},

-- Compute labels for each (entity, seed_time) pair
-- Label window: [seed_time, seed_time + {prediction_window}]
training_labels AS (
    SELECT 
        esp.entity_id AS "EntityID",
        esp.seed_time AS "SeedTime",
        {label_sql} AS "TargetLabel"
    FROM entity_seed_pairs esp{event_join_sql}
    GROUP BY esp.entity_id, esp.seed_time
)

SELECT 
    "EntityID",
    "SeedTime",
    "TargetLabel"
FROM training_labels
ORDER BY "SeedTime", "EntityID"
"""
    
    # Generate CREATE TABLE version if requested
    sql_create_table = None
    if include_create_table:
        sql_create_table = f"""
CREATE TABLE training_table AS
{main_query};

-- Add indexes for efficient querying
CREATE INDEX idx_training_seed_time ON training_table ("SeedTime");
CREATE INDEX idx_training_entity ON training_table ("EntityID");
"""
    
    # Additional notes based on task
    notes.extend([
        f"Prediction window: {prediction_window} ({prediction_window_days} days)",
        f"Slide step: {slide_step} ({slide_step_days} days)",
        f"SQL dialect: {dialect}",
    ])
    
    # Only add placeholder notes if there are still placeholders
    if parameters:
        notes.append("Replace placeholders ({{...}}) with actual table/column names.")
    notes.append("Ensure proper temporal filtering to prevent data leakage in feature computation.")
    
    result = {
        "status": "SQL generated",
        "sql_query": main_query.strip(),
        "sql_create_table": sql_create_table.strip() if sql_create_table else None,
        "parameters": parameters if parameters else None,
        "notes": notes,
        "requires_refinement": requires_refinement,
        "dialect": dialect,
        "metadata_used": {
            "entity_table": entity_table,
            "entity_id_column": entity_id_column,
            "event_table": evt_table,
            "event_timestamp_column": evt_timestamp_col,
            "entity_created_at_column": entity_created_col,
            "prediction_window_days": prediction_window_days,
            "slide_step_days": slide_step_days,
            "active_entity_filtering": entity_created_col is not None
        }
    }
    
    # Store the generated SQL for reference
    object_registry.register(dict, "training_table_sql", result, overwrite=True)
    
    logger.info(f"SQL implementation generated for dialect={dialect}, requires_refinement={requires_refinement}, active_entity_filtering={entity_created_col is not None}")
    
    return result


def _generate_label_sql_template(target_definition: str, entity_id_column: str, config: dict) -> str:
    """
    Generate a SQL template for label computation based on natural language target definition.
    
    This is a heuristic-based template generator. For production use, this should be
    replaced with LLM-based SQL generation.
    """
    target_lower = target_definition.lower()
    
    # Common patterns
    if any(word in target_lower for word in ["purchase", "buy", "order", "transaction"]):
        return "CASE WHEN COUNT(evt.{}) > 0 THEN 1 ELSE 0 END".format(entity_id_column)
    
    elif any(word in target_lower for word in ["churn", "leave", "cancel", "unsubscribe"]):
        return "CASE WHEN COUNT(evt.{}) = 0 THEN 1 ELSE 0 END".format(entity_id_column)
    
    elif any(word in target_lower for word in ["click", "view", "visit", "engage"]):
        return "CASE WHEN COUNT(evt.{}) > 0 THEN 1 ELSE 0 END".format(entity_id_column)
    
    elif any(word in target_lower for word in ["amount", "sum", "total", "revenue"]):
        return "COALESCE(SUM(evt.amount), 0)"
    
    elif any(word in target_lower for word in ["count", "frequency", "times"]):
        return "COUNT(evt.{})".format(entity_id_column)
    
    else:
        # Generic binary classification template
        return f"""
        -- TODO: Refine this label logic based on: {target_definition}
        CASE 
            WHEN COUNT(evt.{entity_id_column}) > 0 THEN 1 
            ELSE 0 
        END"""


@tool
def temporal_split(
    training_table: Dict[str, Any], 
    val_timestamp: str, 
    test_timestamp: str,
    window_size_days: int = 0
) -> Dict[str, Any]:
    """
    Validates and records a temporal split strategy for the Training Table, ensuring NO DATA LEAKAGE.
    
    This tool validates chronological ordering, checks for label leakage due to prediction windows,
    and ensures timestamps are within the actual data range. The actual data splitting should be 
    done via SQL queries using execute_sql_query or generate_temporal_splits_from_db.

    CRITICAL: For temporal prediction tasks with a prediction horizon (e.g., "predict if user 
    contributes in next 2 days"), you MUST set window_size_days to prevent label leakage.
    The gap between val_timestamp and test_timestamp must be >= window_size_days.

    Args:
        training_table: The training table metadata from define_training_task
        val_timestamp: The cutoff timestamp for validation inputs (format: YYYY-MM-DD). 
                       Validation features use data <= val_timestamp, labels use data in 
                       [val_timestamp, val_timestamp + window_size_days].
        test_timestamp: The cutoff timestamp for test inputs (format: YYYY-MM-DD).
                        Test features use data <= test_timestamp.
        window_size_days: The prediction window size in days (delta). For tasks like 
                          "predict activity in next N days", set this to N. Defaults to 0 
                          for instantaneous prediction tasks. This ensures val labels don't 
                          overlap with test data.

    Returns:
        A dictionary containing the split configuration, validation results, and any warnings.
    """
    object_registry = ObjectRegistry()
    
    # Get date ranges from training table - support both old and new format
    # New format from define_training_task has metadata.available_date_range
    # Old format has data_min_date/data_max_date directly
    if "metadata" in training_table:
        # New format from define_training_task
        date_range = training_table.get("metadata", {}).get("available_date_range", {})
        data_min_date = date_range.get("min")
        data_max_date = date_range.get("max")
    else:
        # Legacy format
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
    
    warnings = []
    errors = []
    
    # 1. CRITICAL: Validate Chronological Order
    if val_dt >= test_dt:
        errors.append(
            f"Invalid configuration: val_timestamp ({val_timestamp}) must be strictly "
            f"before test_timestamp ({test_timestamp})."
        )
        logger.error(f"Temporal split error: val_timestamp >= test_timestamp")
        return {
            "status": "error",
            "message": errors[0],
            "val_cutoff": val_timestamp,
            "test_cutoff": test_timestamp,
        }
    
    # 2. CRITICAL: Validate Window Overlap (Leakage Check)
    # Per RDL paper (Appendix C.1), labels are computed from t to t + delta.
    # If test starts at t_test, then val_timestamp + window must not exceed t_test.
    window_delta = timedelta(days=window_size_days)
    gap_days = (test_dt - val_dt).days
    
    if val_dt + window_delta > test_dt:
        leakage_msg = (
            f"DATA LEAKAGE RISK: The prediction window ({window_size_days} days) causes "
            f"validation labels (computed from {val_timestamp} to {val_dt + window_delta}) "
            f"to overlap with test input data (starting at {test_timestamp}). "
            f"Current gap is {gap_days} days, but window requires at least {window_size_days} days. "
            f"Either increase the gap between val and test timestamps, or reduce window_size_days."
        )
        warnings.append(leakage_msg)
        logger.warning(f"Temporal split leakage warning: {leakage_msg}")
    
    # Helper function to parse database dates
    def parse_date(date_str):
        if isinstance(date_str, str):
            # Try different formats
            for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    return datetime.strptime(date_str.split("+")[0].split("Z")[0], fmt)
                except ValueError:
                    continue
        return date_str if isinstance(date_str, datetime) else None
    
    # 3. Validate against actual data range
    db_min = None
    db_max = None
    if data_min_date and data_max_date:
        try:
            db_min = parse_date(data_min_date)
            db_max = parse_date(data_max_date)
            
            if db_min and db_max:
                # Check if val_timestamp is within data range
                if val_dt < db_min:
                    warnings.append(
                        f"val_timestamp {val_timestamp} is before data start date {db_min.date()}. "
                        f"No training data will be available before this timestamp."
                    )
                if val_dt > db_max:
                    warnings.append(
                        f"val_timestamp {val_timestamp} is after data end date {db_max.date()}. "
                        f"No validation labels can be computed."
                    )
                
                # Check if test_timestamp is within data range
                if test_dt < db_min:
                    warnings.append(
                        f"test_timestamp {test_timestamp} is before data start date {db_min.date()}."
                    )
                if test_dt > db_max:
                    warnings.append(
                        f"test_timestamp {test_timestamp} is after data end date {db_max.date()}. "
                        f"Test labels may be incomplete or unavailable."
                    )
                
                # Check if there's enough data for the prediction window after test
                if window_size_days > 0 and test_dt + window_delta > db_max:
                    warnings.append(
                        f"Test prediction window extends beyond data range: "
                        f"test_timestamp + window ({test_dt + window_delta}) > data_max ({db_max.date()}). "
                        f"Test labels may be incomplete."
                    )
                
                if warnings:
                    logger.warning(f"Temporal split warnings: {warnings}")
                    
        except Exception as e:
            logger.warning(f"Could not validate timestamps against data range: {e}")
            warnings.append(f"Could not fully validate timestamps against data range: {str(e)}")
    else:
        warnings.append(
            "Could not validate timestamps against data range - data range not available. "
            "Use discover_temporal_columns first to get data range information."
        )
    
    # Determine status based on warnings
    if any("LEAKAGE" in w for w in warnings):
        status = "Split configuration recorded with CRITICAL WARNINGS - potential data leakage detected."
    elif warnings:
        status = "Split configuration recorded with warnings."
    else:
        status = "Split configuration recorded successfully."
    
    result = {
        "val_cutoff": val_timestamp,
        "test_cutoff": test_timestamp,
        "window_size_days": window_size_days,
        "gap_between_splits_days": gap_days,
        "status": status,
        "warnings": warnings if warnings else None,
        "data_range": {
            "min": data_min_date,
            "max": data_max_date
        } if data_min_date and data_max_date else None,
        "split_summary": {
            "train": f"features: data <= {val_timestamp}, labels: data in [{val_timestamp}, {val_timestamp} + {window_size_days}d]" if window_size_days > 0 else f"data < {val_timestamp}",
            "val": f"features: data <= {val_timestamp}, labels: [{val_timestamp}, {(val_dt + window_delta).strftime('%Y-%m-%d')}]" if window_size_days > 0 else f"data in [{val_timestamp}, {test_timestamp})",
            "test": f"features: data <= {test_timestamp}, labels: [{test_timestamp}, {(test_dt + window_delta).strftime('%Y-%m-%d')}]" if window_size_days > 0 else f"data >= {test_timestamp}",
        },
        "next_steps": "Use execute_sql_query or generate_temporal_splits_from_db to create the actual train/val/test datasets."
    }

    # Register the split info for use by other tools
    object_registry.register(dict, "temporal_split_info", result, overwrite=True)
    
    logger.info(f"Temporal split configured: val={val_timestamp}, test={test_timestamp}, window={window_size_days}d, gap={gap_days}d")

    return result


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
            if entity_id_column in labels_df.columns and entity_id_column in features_df.columns:
                df = features_df.merge(labels_df, on=entity_id_column, how="inner")
            else:
                # Fallback: concat (unsafe, may have misaligned rows)
                logger.warning(f"Entity column '{entity_id_column}' not found in both dataframes. Using concat.")
                df = pd.concat([features_df, labels_df], axis=1)
            
            # Add metadata columns (important for Temporal GNN training)
            df["seed_time"] = seed_time
            df["seed_time_str"] = seed_time.strftime("%Y-%m-%d")
            df["split"] = split_name
            
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
                train_dfs.append(snapshot)
                train_window_dates.append(current_seed.strftime("%Y-%m-%d"))
                logger.debug(f"  Window {i+1}: seed={current_seed.date()}, samples={len(snapshot)}")
            else:
                logger.warning(f"  Window {i+1}: seed={current_seed.date()}, NO DATA")
            
            # Move backwards in time for next window
            current_seed -= timedelta(days=train_stride_days)
        
        # Combine all training windows
        train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        logger.info(f"Training set: {len(train_df)} samples from {len(train_dfs)} windows")
        
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
            for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
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
    
    # Validate window gap to prevent data leakage
    gap_days = (test_cutoff - val_cutoff).days
    warnings = []
    
    if gap_days < window_days:
        leakage_msg = (
            f"DATA LEAKAGE RISK: Gap between val_cutoff and test_cutoff is {gap_days} days, "
            f"but prediction window is {window_days} days. Validation labels will overlap with test data. "
            f"Adjusting test_cutoff to ensure at least {window_days} days gap."
        )
        warnings.append(leakage_msg)
        logger.warning(leakage_msg)
        # Auto-adjust test_cutoff to prevent leakage
        test_cutoff = val_cutoff + timedelta(days=window_days)
        gap_days = window_days
    
    logger.info(f"Validation cutoff: {val_cutoff}, Test cutoff: {test_cutoff}, Gap: {gap_days} days")
    
    # Calculate training window dates (Sliding Window - moving backwards)
    train_stride_days = 30  # Default stride
    train_window_dates = []
    current_seed = val_cutoff - timedelta(days=window_days)  # Start of first training window
    for i in range(num_train_windows):
        train_window_dates.append(str(current_seed.date()))
        current_seed -= timedelta(days=train_stride_days)
    
    # Build split info (consistent with temporal_split output format)
    split_info = {
        "data_range": {
            "min": str(min_date),
            "max": str(max_date)
        },
        "train_end_date": str((val_cutoff - timedelta(days=window_days)).date()),  # Upper bound for training
        "train_window_dates": train_window_dates,  # Each window's seed date
        "val_cutoff": str(val_cutoff.date()),
        "test_cutoff": str(test_cutoff.date()),
        "window_size_days": window_days,  # Renamed for consistency with temporal_split
        "gap_between_splits_days": gap_days,  # Added for consistency
        "num_train_windows": num_train_windows,
        "train_stride_days": train_stride_days,
        "temporal_tables": list(temporal_info.get("tables", {}).keys()),
        "status": "Temporal structure discovered." if not warnings else "Temporal structure discovered with warnings.",
        "warnings": warnings if warnings else None,
        "split_summary": {
            "train": f"Sliding Window: {num_train_windows} windows from {train_window_dates[-1] if train_window_dates else 'N/A'} to {train_window_dates[0] if train_window_dates else 'N/A'}, stride={train_stride_days}d, window={window_days}d",
            "val": f"Single snapshot at {val_cutoff.date()}, labels: [{val_cutoff.date()}, {(val_cutoff + timedelta(days=window_days)).date()}]",
            "test": f"Single snapshot at {test_cutoff.date()}, labels: [{test_cutoff.date()}, {(test_cutoff + timedelta(days=window_days)).date()}]",
        },
        "next_steps": "Use create_temporal_dataset with num_train_windows and train_stride_days params for Sliding Window sampling."
    }
    
    # Register the split info
    object_registry.register(dict, "temporal_split_info", split_info, overwrite=True)
    
    result = {
        "split_info": split_info,
        "registered_datasets": [],
        "exported_files": [],
        "warnings": warnings if warnings else None,
        "message": "Temporal structure analyzed. Data spans from {} to {}. Splits: train until {}, validate until {}, test after {}. Gap: {} days (window: {} days).".format(
            min_date.date(), max_date.date(), val_cutoff.date(), test_cutoff.date(), test_cutoff.date(), gap_days, window_days
        )
    }
    
    return result
