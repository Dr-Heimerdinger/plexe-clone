import os
import logging
from typing import Optional
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

