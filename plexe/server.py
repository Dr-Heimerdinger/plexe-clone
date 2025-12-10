"""
FastAPI server for the Plexe conversational agent.

This module provides a lightweight WebSocket API for the conversational agent
and serves the assistant-ui frontend for local execution.
"""

import asyncio
import contextvars
import functools
import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from plexe.agents.conversational import ConversationalAgent
from plexe.api import datasets_router
from plexe.execution_context import set_chain_of_thought_callable, reset_chain_of_thought_callable
from plexe.internal.common.utils.chain_of_thought import (
    ChainOfThoughtCallable,
    MultiEmitter,
    ConsoleEmitter,
    WebSocketEmitter,
)

logger = logging.getLogger(__name__)


class WebSocketLogHandler(logging.Handler):
    """
    Log handler that forwards log records to a WebSocketEmitter.
    """
    def __init__(self, emitter: WebSocketEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Use the logger name as the agent name, or a simplified version
            agent_name = record.name.split('.')[-1] if '.' in record.name else record.name
            
            # Emit as a thought
            self.emitter.emit_thought(agent_name, msg)
        except Exception:
            self.handleError(record)


app = FastAPI(title="Plexe Assistant", version="1.0.0")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include dataset API routes
app.include_router(datasets_router)

# Serve static files from the ui directory
ui_dir = Path(__file__).parent / "ui"
# Prefer a built frontend in `ui/frontend/dist` if present (Vite build output).
frontend_dist = ui_dir / "frontend" / "dist"
if frontend_dist.exists():
    # Serve built static assets from the frontend dist directory
    app.mount("/static", StaticFiles(directory=str(frontend_dist)), name="static")
elif ui_dir.exists():
    # Fallback: serve legacy ui directory (contains index.html and inline JS)
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    # Prefer serving the built frontend if available
    built_index = frontend_dist / "index.html"
    legacy_index = ui_dir / "index.html"

    if built_index.exists():
        return FileResponse(str(built_index))
    if legacy_index.exists():
        return FileResponse(str(legacy_index))
    return {
        "error": "Frontend not found. Please ensure plexe/ui/frontend/dist/index.html or plexe/ui/index.html exists."
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"New WebSocket connection: {session_id}")

    # Get the current event loop to pass to the emitter
    loop = asyncio.get_running_loop()

    # Create emitters for chain of thought
    ws_emitter = WebSocketEmitter(websocket, loop=loop)
    console_emitter = ConsoleEmitter()
    multi_emitter = MultiEmitter([ws_emitter, console_emitter])
    
    # Create chain of thought callable with the multi emitter
    chain_of_thought = ChainOfThoughtCallable(emitter=multi_emitter)

    # Create a new agent instance for this session with chain of thought
    # Enable verbose mode to capture detailed logs
    agent = ConversationalAgent(chain_of_thought_callable=chain_of_thought, verbose=True)

    # Create and attach log handler to capture agent logs
    log_handler = WebSocketLogHandler(ws_emitter)
    # Capture logs from smolagents and plexe.agents
    loggers_to_capture = ["smolagents", "plexe.agents"]
    for logger_name in loggers_to_capture:
        l = logging.getLogger(logger_name)
        l.addHandler(log_handler)
        # Ensure level is at least INFO to capture thoughts if not already set
        if l.level == logging.NOTSET:
            l.setLevel(logging.INFO)

    # Track if agent is currently running
    agent_task = None

    async def run_agent_task(user_message: str):
        """Run agent in executor and handle response."""
        nonlocal agent_task
        token = set_chain_of_thought_callable(chain_of_thought)
        try:
            ctx = contextvars.copy_context()
            func = functools.partial(ctx.run, agent.agent.run, user_message, reset=False)
            response = await loop.run_in_executor(None, func)
            await websocket.send_json({"role": "assistant", "content": response, "id": str(uuid.uuid4())})
        except Exception as e:
            logger.error(f"Agent error: {e}")
            await websocket.send_json({
                "role": "assistant",
                "content": f"I encountered an error: {str(e)}. Please try again.",
                "id": str(uuid.uuid4()),
                "error": True,
            })
        finally:
            reset_chain_of_thought_callable(token)
            agent_task = None

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                
                # Handle ping messages for keep-alive
                if message_data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                # Handle confirmation response
                if message_data.get("type") == "confirmation_response":
                    request_id = message_data.get("id")
                    confirmed = message_data.get("confirmed", False)
                    logger.info(f"Received confirmation response: {request_id} = {confirmed}")
                    ws_emitter.resolve_confirmation(request_id, confirmed)
                    continue
                
                user_message = message_data.get("content", "")
                if not user_message:
                    continue

                # Process the message with the agent
                logger.debug(f"Processing message: {user_message[:100]}...")
                
                # Start agent task in background so we can continue receiving messages
                # This allows confirmation responses to be processed while agent is running
                if agent_task is None:
                    agent_task = asyncio.create_task(run_agent_task(user_message))
                else:
                    logger.warning("Agent is already processing a message, ignoring new message")
                    await websocket.send_json({
                        "role": "assistant",
                        "content": "I'm still processing your previous request. Please wait.",
                        "id": str(uuid.uuid4()),
                    })

            except json.JSONDecodeError:
                # Handle plain text messages for compatibility
                if agent_task is None:
                    agent_task = asyncio.create_task(run_agent_task(data))

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                try:
                    await websocket.send_json(
                        {
                            "role": "assistant",
                            "content": f"I encountered an error: {str(e)}. Please try again.",
                            "id": str(uuid.uuid4()),
                            "error": True,
                        }
                    )
                except Exception as send_error:
                    logger.warning(f"Could not send error message to client: {send_error}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        if agent_task:
            agent_task.cancel()
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        if agent_task:
            agent_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass # Connection might be already closed
    finally:
        # Remove log handler
        for logger_name in loggers_to_capture:
            logging.getLogger(logger_name).removeHandler(log_handler)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "plexe-assistant"}
