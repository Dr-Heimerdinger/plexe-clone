"""
WebSocket emitter for broadcasting agent chain of thought to web clients.

This emitter sends agent thinking processes and actions to connected WebSocket clients
in real-time, allowing the UI to display the agent's reasoning process.
"""

import asyncio
import logging

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
            loop: The asyncio event loop to use for scheduling messages
        """
        self.websocket = websocket
        self.step_count = 0
        self.loop = loop or asyncio.get_event_loop()

    def emit_thought(self, agent_name: str, message: str) -> None:
        """
        Emit a thought to the WebSocket client.

        Args:
            agent_name: The name of the agent emitting the thought
            message: The thought message
        """
        try:
            self.step_count += 1

            # Prepare the message payload
            payload = {
                "type": "thinking",
                "role": "thinking",
                "agent_name": agent_name,
                "message": message,
                "step_number": self.step_count,
            }

            logger.info(f"WebSocketEmitter: Emitting thought from {agent_name}, step {self.step_count}")

            # Send the message in a thread-safe way
            if self.loop and self.loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self._send_message(payload), self.loop)
                # Wait briefly for the future to complete to catch any errors
                try:
                    future.result(timeout=1.0)
                    logger.info(f"WebSocketEmitter: Successfully sent message for step {self.step_count}")
                except Exception as e:
                    logger.error(f"WebSocketEmitter: Future error: {e}")
            else:
                # Fallback if no loop provided or loop not running (shouldn't happen in this setup)
                logger.warning("WebSocketEmitter: No running loop found, attempting direct async call")
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(self._send_message(payload))
                except RuntimeError:
                    logger.error("WebSocketEmitter: Could not send message - no event loop")

        except Exception as e:
            logger.error(f"Error emitting thought to WebSocket: {e}")

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
