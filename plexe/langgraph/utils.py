"""
Utility functions for LangGraph agents.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def create_working_directory(base_dir: str = "workdir") -> str:
    """
    Create a unique working directory for a session.
    
    Args:
        base_dir: Base directory for workspaces
    
    Returns:
        Path to the created working directory
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(base_dir, f"session-{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    subdirs = ["csv_files", "cache", "artifacts"]
    for subdir in subdirs:
        os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)
    
    return session_dir


def validate_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return os.path.isfile(file_path)


def validate_directory_exists(dir_path: str) -> bool:
    """Check if a directory exists."""
    return os.path.isdir(dir_path)


def get_csv_files_in_directory(dir_path: str) -> list:
    """Get list of CSV files in a directory."""
    if not os.path.isdir(dir_path):
        return []
    return [f for f in os.listdir(dir_path) if f.endswith('.csv')]


def read_file_content(file_path: str) -> Optional[str]:
    """Read content from a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def write_file_content(file_path: str, content: str) -> bool:
    """Write content to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        return False


def format_error_message(error: Exception, context: str = "") -> str:
    """Format an error message with context."""
    msg = f"Error: {type(error).__name__}: {str(error)}"
    if context:
        msg = f"{context}: {msg}"
    return msg


def sanitize_sql_identifier(identifier: str) -> str:
    """Sanitize a SQL identifier to prevent injection."""
    return ''.join(c for c in identifier if c.isalnum() or c == '_')


def format_table_info(tables: Dict[str, Any]) -> str:
    """Format table information for display."""
    lines = []
    for table_name, info in tables.items():
        pk = info.get("primary_key", [])
        cols = [c["name"] for c in info.get("columns", [])]
        lines.append(f"- {table_name}")
        if pk:
            lines.append(f"  PK: {', '.join(pk)}")
        lines.append(f"  Columns: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")
    return "\n".join(lines)


def estimate_task_type(target_description: str) -> str:
    """Estimate task type from description."""
    description_lower = target_description.lower()
    
    binary_indicators = ["churn", "fraud", "click", "convert", "buy", "will", "whether"]
    regression_indicators = ["count", "amount", "price", "revenue", "quantity", "how many"]
    multiclass_indicators = ["category", "class", "type", "segment", "which"]
    
    for indicator in binary_indicators:
        if indicator in description_lower:
            return "binary_classification"
    
    for indicator in regression_indicators:
        if indicator in description_lower:
            return "regression"
    
    for indicator in multiclass_indicators:
        if indicator in description_lower:
            return "multiclass_classification"
    
    return "regression"


def get_default_metrics(task_type: str) -> list:
    """Get default metrics for a task type."""
    metrics_map = {
        "regression": ["mae", "rmse", "r2"],
        "binary_classification": ["accuracy", "auroc", "f1"],
        "multiclass_classification": ["accuracy", "f1_macro", "f1_micro"],
    }
    return metrics_map.get(task_type, ["mae"])


def validate_python_code(code: str) -> Dict[str, Any]:
    """Validate Python code for syntax errors."""
    try:
        compile(code, '<string>', 'exec')
        return {"valid": True, "errors": []}
    except SyntaxError as e:
        return {
            "valid": False,
            "errors": [{
                "line": e.lineno,
                "offset": e.offset,
                "message": e.msg,
            }]
        }
