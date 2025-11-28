"""
Tools for performing I/O operations, such as reading from and writing to the filesystem.
"""

import logging
import os
from smolagents import tool

from plexe.core.object_registry import ObjectRegistry

logger = logging.getLogger(__name__)


def _get_session_workdir(workdir: str = "workdir") -> str:
    """
    Get the session's working directory.
    """
    try:
        session_id = ObjectRegistry().get(str, "session_id")
    except KeyError:
        # This is a fallback for when session_id is not in the registry.
        # It should not happen in the normal flow.
        session_id = "default_session"
    session_workdir = os.path.join(workdir, session_id)
    if not os.path.exists(session_workdir):
        os.makedirs(session_workdir)
    return session_workdir


@tool
def read_file(file_path: str) -> str:
    """
    Reads the content of a file from the filesystem.

    Args:
        file_path: The path to the file to read.

    Returns:
        The content of the file as a string.
    """
    workdir = _get_session_workdir()
    path = os.path.join(workdir, file_path)
    try:
        with open(path, "r") as f:
            content = f.read()
        logger.debug(f"✅ Read file: {path}")
        return content
    except Exception as e:
        logger.error(f"🔥 Error reading file {path}: {e}")
        raise


@tool
def write_file(file_path: str, content: str) -> str:
    """
    Writes content to a file in the filesystem.

    Args:
        file_path: The path to the file to write to.
        content: The content to write to the file.

    Returns:
        The path to the written file.
    """
    workdir = _get_session_workdir()
    path = os.path.join(workdir, file_path)
    try:
        with open(path, "w") as f:
            f.write(content)
        logger.debug(f"✅ Wrote to file: {path}")
        return path
    except Exception as e:
        logger.error(f"🔥 Error writing to file {path}: {e}")
        raise
