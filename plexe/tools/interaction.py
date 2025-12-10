from smolagents import tool
from plexe.execution_context import get_chain_of_thought_callable
from plexe.internal.common.utils.chain_of_thought.websocket_emitter import WebSocketEmitter
from plexe.internal.common.utils.chain_of_thought.emitters import MultiEmitter
from plexe.core.object_registry import ObjectRegistry
import os
import logging

logger = logging.getLogger(__name__)


def _get_working_dir() -> str:
    """Get working directory from registry or use default."""
    try:
        return ObjectRegistry().get(str, "working_dir")
    except KeyError:
        return "./workdir/"


@tool
def save_to_workdir(filename: str, content: str, subfolder: str = "") -> str:
    """
    Saves content to a file in the working directory.
    Use this tool to persist generated code, data summaries, or any other artifacts.
    
    Args:
        filename: The name of the file to save (e.g., 'training_code.py', 'data_summary.txt')
        content: The content to write to the file
        subfolder: Optional subfolder within workdir (e.g., 'code', 'data')
        
    Returns:
        str: The full path where the file was saved, or error message
    """
    try:
        working_dir = _get_working_dir()
        
        if subfolder:
            save_dir = os.path.join(working_dir, subfolder)
        else:
            save_dir = working_dir
            
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Saved file to {file_path}")
        return f"Successfully saved to {file_path}"
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        return f"Error saving file: {e}"


@tool
def ask_user_confirmation(title: str, content: str, content_type: str = "text", file_path: str = None) -> str:
    """
    Pauses execution and asks the user for confirmation via the UI.
    Also saves the content to a file for debugging.
    
    Args:
        title: The title of the confirmation dialog.
        content: The content to display to the user.
        content_type: The type of content ('text', 'code', 'json', 'markdown').
        file_path: Optional path to save the content to (for debugging).
        
    Returns:
        str: "Confirmed" if the user confirmed, or "Rejected" if rejected.
    """
    # Save to file if requested
    if file_path:
        try:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
            logger.info(f"Saved content to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save content to {file_path}: {e}")

    # Get the emitter
    cot_callable = get_chain_of_thought_callable()
    if not cot_callable or not hasattr(cot_callable, 'emitter'):
        logger.warning("No chain of thought emitter found. Assuming confirmation.")
        return "Confirmed (Auto - No UI)"
        
    emitter = cot_callable.emitter
    
    # Handle MultiEmitter
    ws_emitter = None
    if isinstance(emitter, MultiEmitter):
        for e in emitter.emitters:
            if isinstance(e, WebSocketEmitter):
                ws_emitter = e
                break
    elif isinstance(emitter, WebSocketEmitter):
        ws_emitter = emitter
        
    if not ws_emitter:
        logger.warning("No WebSocket emitter found. Assuming confirmation.")
        return "Confirmed (Auto - No UI)"
        
    # Request confirmation
    logger.info(f"Requesting confirmation: {title}")
    confirmed = ws_emitter.request_confirmation(title, content, content_type)
    
    if confirmed:
        return "Confirmed"
    else:
        return "Rejected"
