from typing import Dict, Any
from langchain_core.tools import tool as langchain_tool

@langchain_tool
def get_dataset_preview(
    dataset_path: str,
    num_rows: int = 5
) -> Dict[str, Any]:
    """
    Preview a dataset by showing the first few rows and schema information.
    
    Args:
        dataset_path: Path to the CSV file or directory containing CSV files
        num_rows: Number of rows to preview (default: 5)
    
    Returns:
        Dictionary with schema info and sample data
    """
    import pandas as pd
    import os
    
    result = {"tables": {}, "total_tables": 0}
    
    if os.path.isdir(dataset_path):
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        result["total_tables"] = len(csv_files)
        
        for csv_file in csv_files[:10]:
            table_name = csv_file.replace('.csv', '')
            file_path = os.path.join(dataset_path, csv_file)
            try:
                df = pd.read_csv(file_path, nrows=num_rows)
                result["tables"][table_name] = {
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "sample_data": df.to_dict(orient='records'),
                    "row_count": len(pd.read_csv(file_path)),
                }
            except Exception as e:
                result["tables"][table_name] = {"error": str(e)}
    else:
        try:
            df = pd.read_csv(dataset_path, nrows=num_rows)
            table_name = os.path.basename(dataset_path).replace('.csv', '')
            result["tables"][table_name] = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_data": df.to_dict(orient='records'),
                "row_count": len(pd.read_csv(dataset_path)),
            }
            result["total_tables"] = 1
        except Exception as e:
            result["error"] = str(e)
    
    return result
