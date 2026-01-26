from typing import Dict, Any

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
