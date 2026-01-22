"""
Common tools shared across multiple agents.
"""

from typing import Dict
from langchain_core.tools import tool as langchain_tool


@langchain_tool
def save_artifact(
    content: str,
    filename: str,
    working_dir: str
) -> Dict[str, str]:
    """
    Save an artifact file to the working directory.
    
    Args:
        content: Content to save
        filename: Name of the file
        working_dir: Working directory
    
    Returns:
        Save status and file path
    """
    import os
    
    os.makedirs(working_dir, exist_ok=True)
    file_path = os.path.join(working_dir, filename)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    return {
        "status": "saved",
        "file_path": file_path
    }
