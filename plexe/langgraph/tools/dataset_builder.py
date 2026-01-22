"""
Tools for the Dataset Builder Agent.
"""

from typing import Dict, Any
from langchain_core.tools import tool as langchain_tool


@langchain_tool
def get_csv_files_info(csv_dir: str) -> Dict[str, Any]:
    """
    Get information about CSV files in a directory.
    
    Args:
        csv_dir: Directory containing CSV files
    
    Returns:
        Dictionary with file information
    """
    import os
    import pandas as pd
    
    files = []
    for f in os.listdir(csv_dir):
        if f.endswith('.csv'):
            file_path = os.path.join(csv_dir, f)
            try:
                df = pd.read_csv(file_path, nrows=1)
                row_count = sum(1 for _ in open(file_path)) - 1
                files.append({
                    "name": f.replace('.csv', ''),
                    "path": file_path,
                    "columns": list(df.columns),
                    "row_count": row_count
                })
            except Exception as e:
                files.append({"name": f, "error": str(e)})
    
    return {"files": files, "count": len(files)}


@langchain_tool
def get_temporal_statistics(csv_dir: str) -> Dict[str, Any]:
    """
    Analyze temporal columns in CSV files to determine val/test timestamps.
    
    Args:
        csv_dir: Directory containing CSV files
    
    Returns:
        Dictionary with temporal analysis and suggested timestamps
    """
    import pandas as pd
    import os
    
    temporal_stats = {}
    all_timestamps = []
    
    for f in os.listdir(csv_dir):
        if not f.endswith('.csv'):
            continue
        
        table_name = f.replace('.csv', '')
        file_path = os.path.join(csv_dir, f)
        
        try:
            df = pd.read_csv(file_path)
            table_temporal = {}
            
            for col in df.columns:
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    valid_count = parsed.notna().sum()
                    if valid_count > len(df) * 0.5:
                        min_ts = parsed.min()
                        max_ts = parsed.max()
                        table_temporal[col] = {
                            "min": str(min_ts),
                            "max": str(max_ts),
                            "valid_count": int(valid_count)
                        }
                        all_timestamps.extend(parsed.dropna().tolist())
                except:
                    pass
            
            if table_temporal:
                temporal_stats[table_name] = table_temporal
        except Exception as e:
            temporal_stats[table_name] = {"error": str(e)}
    
    suggested_splits = {}
    if all_timestamps:
        all_timestamps = sorted(all_timestamps)
        n = len(all_timestamps)
        suggested_splits = {
            "val_timestamp": str(all_timestamps[int(n * 0.7)]),
            "test_timestamp": str(all_timestamps[int(n * 0.85)]),
        }
    
    return {
        "temporal_stats": temporal_stats,
        "suggested_splits": suggested_splits
    }


@langchain_tool
def register_dataset_code(
    code: str,
    class_name: str,
    file_path: str
) -> Dict[str, str]:
    """
    Register generated Dataset class code.
    
    Args:
        code: Python code for the Dataset class
        class_name: Name of the Dataset class
        file_path: Path where the code will be saved
    
    Returns:
        Registration status
    """
    import os
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write(code)
    
    return {
        "status": "registered",
        "class_name": class_name,
        "file_path": file_path,
        "code": code
    }
