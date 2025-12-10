"""
WebSocket emitter for broadcasting agent chain of thought to web clients.

This emitter sends agent thinking processes and actions to connected WebSocket clients
in real-time, allowing the UI to display the agent's reasoning process.
"""

import asyncio
import logging
import threading
import uuid

from plexe.internal.common.utils.chain_of_thought.emitters import ChainOfThoughtEmitter

logger = logging.getLogger(__name__)


class WebSocketEmitter(ChainOfThoughtEmitter):
    """
    Emitter that broadcasts chain of thought to WebSocket clients.
    
    This emitter sends agent thinking messages to a WebSocket connection,
    allowing real-time display of agent reasoning in the UI.
    """

    def __init__(self, websocket, loop=None):
        """
        Initialize the WebSocket emitter.

        Args:
            websocket: The FastAPI WebSocket instance to send messages to
            loop: The asyncio event loop to use for sending messages (optional)
        """
        self.websocket = websocket
        self.step_count = 0
        self._message_queue = []
        self._loop = loop
        self._lock = threading.Lock()
        self._pending_confirmations = {}
        self._confirmation_results = {}

    def set_loop(self, loop):
        """Set the event loop to use for sending messages."""
        self._loop = loop

    def request_confirmation(self, title: str, content: str, content_type: str = "text") -> bool:
        """
        Send a confirmation request to the UI and wait for response.
        
        Args:
            title: Title of the confirmation dialog
            content: Content to display
            content_type: Type of content (text, code, json, markdown)
            
        Returns:
            bool: True if confirmed, False otherwise
        """
        request_id = str(uuid.uuid4())
        event = threading.Event()
        
        with self._lock:
            self._pending_confirmations[request_id] = event
            
        payload = {
            "type": "confirmation_request",
            "id": request_id,
            "title": title,
            "content": content,
            "content_type": content_type
        }
        
        self._emit_payload(payload)
        
        logger.info(f"Waiting for confirmation {request_id}...")
        event.wait()
        
        with self._lock:
            result = self._confirmation_results.pop(request_id, False)
            if request_id in self._pending_confirmations:
                del self._pending_confirmations[request_id]
            
        return result

    def resolve_confirmation(self, request_id: str, confirmed: bool):
        """Resolve a pending confirmation request."""
        with self._lock:
            if request_id in self._pending_confirmations:
                self._confirmation_results[request_id] = confirmed
                self._pending_confirmations[request_id].set()
            else:
                logger.warning(f"Received confirmation for unknown request {request_id}")

    def emit_thought(self, agent_name: str, message: str) -> None:
        """
        Emit a thought to the WebSocket client.

        Args:
            agent_name: The name of the agent emitting the thought
            message: The thought message
        """
        try:
            with self._lock:
                self.step_count += 1
                step_num = self.step_count
            
            # Prepare the message payload
            payload = {
                "type": "thinking",
                "role": "thinking",
                "agent_name": agent_name,
                "message": message,
                "step_number": step_num,
            }
            
            self._emit_payload(payload)
                
        except Exception as e:
            logger.error(f"Error emitting thought to WebSocket: {e}")

    def _emit_payload(self, payload: dict):
        """Internal helper to send payload via WebSocket."""
        try:
            # Try to send the message
            # First, check if we're in the same thread as the event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, can create task directly
                asyncio.create_task(self._send_message(payload))
            except RuntimeError:
                # Not in async context - we're likely in a thread pool
                # Use run_coroutine_threadsafe if we have a loop reference
                if self._loop is not None and self._loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        self._send_message(payload), 
                        self._loop
                    )
                    # Don't wait for result to avoid blocking
                    # But add a callback to log errors
                    future.add_done_callback(self._handle_send_result)
                else:
                    # Last resort: queue the message
                    logger.warning("WebSocketEmitter: No event loop available - message queued")
                    self._message_queue.append(payload)
        except Exception as e:
            logger.error(f"Error emitting payload to WebSocket: {e}")


    def _handle_send_result(self, future):
        """Handle the result of a threadsafe coroutine call."""
        try:
            future.result()  # This will raise if there was an error
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")

    async def _send_message(self, payload: dict) -> None:
        """
        Send a message to the WebSocket client.
        
        Args:
            payload: The message payload to send
        """
        try:
            await self.websocket.send_json(payload)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")

