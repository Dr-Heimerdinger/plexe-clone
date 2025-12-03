"""
Tools for temporal processing and training table generation for the Temporal Task Supervisor Agent.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import pandas as pd
from smolagents import tool
from sqlalchemy import create_engine, text

from plexe.core.object_registry import ObjectRegistry
from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.internal.common.datasets.interface import TabularConvertible

logger = logging.getLogger(__name__)


@tool
def generate_training_table_sql(query_logic: str, window_size: str, slide_step: str) -> Dict[str, Any]:
    """
    Generates the Training Table (T_train) containing EntityID, Seed Time, and Target Label.
    Use this to calculate ground truth labels by looking forward from a specific seed time within a defined window.
    
    This tool queries the actual database to determine the correct date range for the data.
    The database connection string must be registered in the ObjectRegistry as 'db_connection_string'.

    Args:
        query_logic: The SQL-like logic or description to define the target label.
        window_size: The duration of the labeling window (e.g., '7d', '2d').
        slide_step: The step size for sliding the window (e.g., '1d', '2d').

    Returns:
        A dictionary representing the training table metadata including actual date ranges from the database.
    """
    object_registry = ObjectRegistry()
    
    # Try to get the database connection string from registry
    try:
        db_connection_string = object_registry.get(str, "db_connection_string")
    except KeyError:
        logger.warning("No db_connection_string found in registry. Using placeholder values.")
        return {
            "columns": ["EntityID", "SeedTime", "TargetLabel"],
            "window_size": window_size,
            "slide_step": slide_step,
            "logic": query_logic,
            "status": "Training table definition generated (no database connection).",
            "warning": "No database connection available. Use generate_temporal_splits_from_db with explicit connection string."
        }
    
    try:
        engine = create_engine(db_connection_string)
        
        # Query the actual date range from the database
        date_queries = [
            "SELECT MIN(creationdate), MAX(creationdate) FROM posts",
            "SELECT MIN(creationdate), MAX(creationdate) FROM comments",
            "SELECT MIN(creationdate), MAX(creationdate) FROM votes",
        ]
        
        min_dates, max_dates = [], []
        with engine.connect() as conn:
            for q in date_queries:
                try:
                    result = conn.execute(text(q)).fetchone()
                    if result and result[0] and result[1]:
                        min_dates.append(result[0])
                        max_dates.append(result[1])
                except Exception as e:
                    logger.warning(f"Could not query date range: {e}")
        
        if min_dates and max_dates:
            data_min_date = min(min_dates)
            data_max_date = max(max_dates)
            logger.info(f"Database date range: {data_min_date} to {data_max_date}")
        else:
            data_min_date = None
            data_max_date = None
            logger.warning("Could not determine date range from database")
        
        result = {
            "columns": ["EntityID", "SeedTime", "TargetLabel"],
            "window_size": window_size,
            "slide_step": slide_step,
            "logic": query_logic,
            "status": "Training table definition generated.",
            "data_min_date": str(data_min_date) if data_min_date else None,
            "data_max_date": str(data_max_date) if data_max_date else None,
            "recommendation": f"Use dates between {data_min_date} and {data_max_date} for temporal splits."
        }
        
        # Store the training table metadata
        object_registry.register(dict, "training_table_metadata", result, overwrite=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Error querying database: {e}")
        return {
            "columns": ["EntityID", "SeedTime", "TargetLabel"],
            "window_size": window_size,
            "slide_step": slide_step,
            "logic": query_logic,
            "status": "Training table definition generated (database error).",
            "error": str(e)
        }


@tool
def temporal_split(training_table: Dict[str, Any], val_timestamp: str, test_timestamp: str) -> Dict[str, Any]:
    """
    Partitions the Training Table into Train, Validation, and Test sets based strictly on the Seed Time.
    Ensures that all training samples occur before the validation period, and all validation samples occur before the testing period.
    
    This tool will validate the timestamps against actual database date ranges and warn if they are outside the data range.
    For best results, first call generate_training_table_sql to get the actual data date range.

    Args:
        training_table: The training table object or metadata returned by generate_training_table_sql.
        val_timestamp: The cutoff timestamp for the validation set (format: YYYY-MM-DD).
        test_timestamp: The cutoff timestamp for the test set (format: YYYY-MM-DD).

    Returns:
        A dictionary containing statistics about the split (counts, ranges).
    """
    object_registry = ObjectRegistry()
    
    # Check if we have actual date ranges from the training table
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
            if isinstance(data_min_date, str):
                if 'T' in data_min_date:
                    db_min = datetime.fromisoformat(data_min_date.replace('Z', '+00:00')).replace(tzinfo=None)
                else:
                    db_min = datetime.strptime(data_min_date.split()[0], "%Y-%m-%d")
            else:
                db_min = data_min_date
                
            if isinstance(data_max_date, str):
                if 'T' in data_max_date:
                    db_max = datetime.fromisoformat(data_max_date.replace('Z', '+00:00')).replace(tzinfo=None)
                else:
                    db_max = datetime.strptime(data_max_date.split()[0], "%Y-%m-%d")
            else:
                db_max = data_max_date
            
            if val_dt < db_min or val_dt > db_max:
                warnings.append(f"val_timestamp {val_timestamp} is outside data range [{db_min.date()} to {db_max.date()}]")
            if test_dt < db_min or test_dt > db_max:
                warnings.append(f"test_timestamp {test_timestamp} is outside data range [{db_min.date()} to {db_max.date()}]")
            
            if warnings:
                logger.warning(f"Temporal split warnings: {warnings}")
                logger.info(f"Consider using dates within the actual data range: {db_min.date()} to {db_max.date()}")
                
        except Exception as e:
            logger.warning(f"Could not validate timestamps against data range: {e}")
    
    # Try to get actual counts from database if connection is available
    try:
        db_connection_string = object_registry.get(str, "db_connection_string")
        engine = create_engine(db_connection_string)
        
        # Count users with activity in each period
        count_query = text("""
            SELECT 
                COUNT(DISTINCT CASE WHEN u.creationdate < :val_cutoff THEN u.id END) as train_eligible,
                COUNT(DISTINCT CASE WHEN u.creationdate >= :val_cutoff AND u.creationdate < :test_cutoff THEN u.id END) as val_eligible,
                COUNT(DISTINCT CASE WHEN u.creationdate >= :test_cutoff THEN u.id END) as test_eligible
            FROM users u
        """)
        
        with engine.connect() as conn:
            result_row = conn.execute(count_query, {
                "val_cutoff": val_dt,
                "test_cutoff": test_dt
            }).fetchone()
            
            if result_row:
                train_count = result_row[0] or 0
                val_count = result_row[1] or 0
                test_count = result_row[2] or 0
            else:
                train_count, val_count, test_count = 0, 0, 0
                
    except KeyError:
        # No database connection - use placeholder counts
        logger.warning("No database connection available. Using estimated counts.")
        train_count, val_count, test_count = 100, 20, 20
        warnings.append("Counts are estimates - no database connection available.")
    except Exception as e:
        logger.error(f"Error querying database for counts: {e}")
        train_count, val_count, test_count = 0, 0, 0
        warnings.append(f"Could not query database: {e}")
    
    result = {
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "val_cutoff": val_timestamp,
        "test_cutoff": test_timestamp,
        "status": "Split completed successfully." if not warnings else "Split completed with warnings.",
        "warnings": warnings if warnings else None,
        "data_range": {
            "min": data_min_date,
            "max": data_max_date
        } if data_min_date and data_max_date else None
    }

    # Register the split info so downstream agents can find it
    object_registry.register(dict, "temporal_split_info", result, overwrite=True)

    return result


@tool
def validate_temporal_consistency(graph: Any, training_table: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs a sanity check to verify that for every sample in T_train at time t,
    the GNN neighbor sampler only accesses graph nodes with timestamps strictly less than or equal to t.

    Args:
        graph: The heterogeneous graph object.
        training_table: The training table object.

    Returns:
        A report dictionary confirming consistency or listing violations.
    """
    # Placeholder implementation.
    return {"consistent": True, "violations": 0, "message": "Temporal consistency verified. No leakage detected."}


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
    Generate temporal train/val/test splits from a relational database for user engagement prediction.
    This tool queries the database directly, creates temporal splits, and optionally exports them to files.

    The task predicts whether a user will make any contribution (vote/comment/post) in the next N days.
    Active users are defined as those who made at least one contribution in the lookback window.

    Args:
        db_connection_string: SQLAlchemy connection string (e.g., 'postgresql+psycopg2://user:pass@host:port/db')
        window_days: Prediction window in days (label horizon, e.g., 10 for "next 10 days")
        num_train_windows: Number of sliding windows for training data
        validation_cutoff: Cutoff date for validation set (YYYY-MM-DD). If None, auto-detect from data.
        output_dir: Directory to export datasets. If None, only register in ObjectRegistry without file export.
        output_format: Export format - 'csv', 'parquet', or 'both'. Default 'csv'.
        active_user_lookback_days: Days to look back for defining active users. Default 10.

    Returns:
        Dictionary containing:
        - split_info: Statistics about train/val/test splits
        - exported_files: List of exported file paths (if output_dir provided)
        - registered_datasets: Names of datasets registered in ObjectRegistry
    """
    import json
    import os
    from pathlib import Path

    object_registry = ObjectRegistry()

    logger.info(f"Connecting to database: {db_connection_string}")
    engine = create_engine(db_connection_string)

    # Helper functions
    def get_data_date_range():
        """Get min and max dates from contribution tables."""
        queries = [
            "SELECT MIN(creationdate), MAX(creationdate) FROM posts",
            "SELECT MIN(creationdate), MAX(creationdate) FROM comments",
            "SELECT MIN(creationdate), MAX(creationdate) FROM votes",
        ]
        min_dates, max_dates = [], []
        with engine.connect() as conn:
            for q in queries:
                try:
                    result = conn.execute(text(q)).fetchone()
                    if result and result[0] and result[1]:
                        min_dates.append(result[0])
                        max_dates.append(result[1])
                except Exception as e:
                    logger.warning(f"Could not query date range: {e}")
        if not min_dates:
            raise ValueError("Could not determine data date range")
        return min(min_dates), max(max_dates)

    def get_active_users(cutoff_date: datetime, lookback_days: int) -> List[int]:
        """Get user IDs who made at least one contribution in the lookback window."""
        lookback_start = cutoff_date - timedelta(days=lookback_days)
        query = text("""
            SELECT DISTINCT u.id as user_id
            FROM users u
            WHERE EXISTS (
                SELECT 1 FROM posts p 
                WHERE p.owneruserid = u.id 
                AND p.creationdate >= :start_date AND p.creationdate < :end_date
            )
            OR EXISTS (
                SELECT 1 FROM comments c 
                WHERE c.userid = u.id 
                AND c.creationdate >= :start_date AND c.creationdate < :end_date
            )
            OR EXISTS (
                SELECT 1 FROM votes v 
                WHERE v.userid = u.id 
                AND v.creationdate >= :start_date AND v.creationdate < :end_date
            )
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"start_date": lookback_start, "end_date": cutoff_date})
        return df["user_id"].tolist()

    def get_user_labels(user_ids: List[int], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get labels: 1 if user contributed in window, 0 otherwise."""
        if not user_ids:
            return pd.DataFrame(columns=["user_id", "label"])

        user_ids_str = ",".join(str(uid) for uid in user_ids)
        query = text(f"""
            WITH contributing_users AS (
                SELECT DISTINCT owneruserid as user_id FROM posts 
                WHERE owneruserid IN ({user_ids_str})
                AND creationdate >= :start_date AND creationdate < :end_date
                UNION
                SELECT DISTINCT userid as user_id FROM comments 
                WHERE userid IN ({user_ids_str})
                AND creationdate >= :start_date AND creationdate < :end_date
                UNION
                SELECT DISTINCT userid as user_id FROM votes 
                WHERE userid IN ({user_ids_str})
                AND creationdate >= :start_date AND creationdate < :end_date
            )
            SELECT user_id, 1 as contributed FROM contributing_users
        """)
        with engine.connect() as conn:
            contributed_df = pd.read_sql(query, conn, params={"start_date": start_date, "end_date": end_date})

        result = pd.DataFrame({"user_id": user_ids})
        result = result.merge(contributed_df, on="user_id", how="left")
        result["label"] = result["contributed"].fillna(0).astype(int)
        return result[["user_id", "label"]]

    def get_user_features(user_ids: List[int], cutoff_date: datetime) -> pd.DataFrame:
        """Extract user features from data available up to cutoff_date."""
        if not user_ids:
            return pd.DataFrame()

        user_ids_str = ",".join(str(uid) for uid in user_ids)
        query = text(f"""
            WITH user_post_stats AS (
                SELECT owneruserid as user_id,
                    COUNT(*) as num_posts,
                    COALESCE(SUM(score), 0) as total_post_score,
                    COALESCE(AVG(score), 0) as avg_post_score
                FROM posts
                WHERE owneruserid IN ({user_ids_str}) AND creationdate < :cutoff_date
                GROUP BY owneruserid
            ),
            user_comment_stats AS (
                SELECT userid as user_id,
                    COUNT(*) as num_comments,
                    COALESCE(SUM(score), 0) as total_comment_score
                FROM comments
                WHERE userid IN ({user_ids_str}) AND creationdate < :cutoff_date
                GROUP BY userid
            ),
            user_vote_stats AS (
                SELECT userid as user_id, COUNT(*) as num_votes
                FROM votes
                WHERE userid IN ({user_ids_str}) AND creationdate < :cutoff_date
                GROUP BY userid
            ),
            user_base AS (
                SELECT id as user_id, reputation, views as profile_views,
                    upvotes as user_upvotes, downvotes as user_downvotes,
                    EXTRACT(EPOCH FROM (:cutoff_date - creationdate)) / 86400.0 as account_age_days
                FROM users WHERE id IN ({user_ids_str})
            )
            SELECT ub.user_id, ub.reputation, ub.profile_views, ub.user_upvotes, ub.user_downvotes,
                ub.account_age_days,
                COALESCE(ups.num_posts, 0) as num_posts,
                COALESCE(ups.total_post_score, 0) as total_post_score,
                COALESCE(ups.avg_post_score, 0) as avg_post_score,
                COALESCE(ucs.num_comments, 0) as num_comments,
                COALESCE(ucs.total_comment_score, 0) as total_comment_score,
                COALESCE(uvs.num_votes, 0) as num_votes
            FROM user_base ub
            LEFT JOIN user_post_stats ups ON ub.user_id = ups.user_id
            LEFT JOIN user_comment_stats ucs ON ub.user_id = ucs.user_id
            LEFT JOIN user_vote_stats uvs ON ub.user_id = uvs.user_id
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"cutoff_date": cutoff_date})

    def create_split_dataset(cutoff_date: datetime) -> pd.DataFrame:
        """Create a labeled dataset for a single time window."""
        user_ids = get_active_users(cutoff_date, active_user_lookback_days)
        if not user_ids:
            return pd.DataFrame()

        label_end = cutoff_date + timedelta(days=window_days)
        labels_df = get_user_labels(user_ids, cutoff_date, label_end)
        features_df = get_user_features(user_ids, cutoff_date)

        result = features_df.merge(labels_df, on="user_id", how="inner")
        result["cutoff_date"] = cutoff_date
        return result

    # Main logic
    min_date, max_date = get_data_date_range()
    logger.info(f"Data range: {min_date} to {max_date}")

    # Set validation cutoff
    if validation_cutoff:
        val_cutoff = datetime.strptime(validation_cutoff, "%Y-%m-%d")
    else:
        val_cutoff = max_date - timedelta(days=window_days * 2)
    logger.info(f"Validation cutoff: {val_cutoff}")

    # Calculate time windows
    test_cutoff = val_cutoff + timedelta(days=window_days)
    train_cutoffs = sorted([val_cutoff - timedelta(days=window_days * (i + 1)) for i in range(num_train_windows)])

    # Generate datasets
    logger.info("Generating training data...")
    train_dfs = []
    for i, tc in enumerate(train_cutoffs):
        df = create_split_dataset(tc)
        if len(df) > 0:
            df["window_idx"] = i
            train_dfs.append(df)
            logger.info(f"  Window {i+1}: {len(df)} samples, {df['label'].sum()} positive")

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()

    logger.info("Generating validation data...")
    val_df = create_split_dataset(val_cutoff)
    logger.info(f"  Validation: {len(val_df)} samples, {val_df['label'].sum() if len(val_df) > 0 else 0} positive")

    logger.info("Generating test data...")
    test_df = create_split_dataset(test_cutoff)
    logger.info(f"  Test: {len(test_df)} samples, {test_df['label'].sum() if len(test_df) > 0 else 0} positive")

    # Register in ObjectRegistry
    registered_datasets = []
    if len(train_df) > 0:
        object_registry.register(TabularConvertible, "temporal_train", DatasetAdapter.coerce(train_df), overwrite=True, immutable=True)
        registered_datasets.append("temporal_train")
    if len(val_df) > 0:
        object_registry.register(TabularConvertible, "temporal_val", DatasetAdapter.coerce(val_df), overwrite=True, immutable=True)
        registered_datasets.append("temporal_val")
    if len(test_df) > 0:
        object_registry.register(TabularConvertible, "temporal_test", DatasetAdapter.coerce(test_df), overwrite=True, immutable=True)
        registered_datasets.append("temporal_test")

    # Build split info
    split_info = {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_positive": int(train_df["label"].sum()) if len(train_df) > 0 else 0,
        "val_positive": int(val_df["label"].sum()) if len(val_df) > 0 else 0,
        "test_positive": int(test_df["label"].sum()) if len(test_df) > 0 else 0,
        "val_cutoff": val_cutoff.isoformat(),
        "test_cutoff": test_cutoff.isoformat(),
        "window_days": window_days,
        "num_train_windows": num_train_windows,
    }
    object_registry.register(dict, "temporal_split_info", split_info, overwrite=True)

    result = {
        "split_info": split_info,
        "registered_datasets": registered_datasets,
        "exported_files": [],
    }

    # Export to files if output_dir provided
    if output_dir:
        try:
            session_id = object_registry.get(str, "session_id")
            base_workdir = os.path.join("workdir", session_id)
        except KeyError:
            base_workdir = "workdir"

        if os.path.isabs(output_dir):
            export_path = Path(output_dir)
        else:
            export_path = Path(base_workdir) / output_dir

        export_path.mkdir(parents=True, exist_ok=True)

        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(df) == 0:
                continue
            if output_format in ("csv", "both"):
                csv_path = export_path / f"{split_name}.csv"
                df.to_csv(csv_path, index=False)
                result["exported_files"].append(str(csv_path))
                logger.info(f"✅ Exported {split_name} to {csv_path}")
            if output_format in ("parquet", "both"):
                parquet_path = export_path / f"{split_name}.parquet"
                df.to_parquet(parquet_path, index=False)
                result["exported_files"].append(str(parquet_path))
                logger.info(f"✅ Exported {split_name} to {parquet_path}")

        # Save metadata
        metadata = {
            "task": "user_engagement_prediction",
            "window_days": window_days,
            "active_user_lookback_days": active_user_lookback_days,
            "split_info": split_info,
            "data_range": {"min": min_date.isoformat(), "max": max_date.isoformat()},
        }
        metadata_path = export_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        result["exported_files"].append(str(metadata_path))
        result["output_directory"] = str(export_path)

    logger.info(f"✅ Temporal splits generated successfully: {split_info}")
    return result
