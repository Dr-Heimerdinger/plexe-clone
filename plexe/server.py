"""
FastAPI server for the Plexe conversational agent.

This module provides a lightweight WebSocket API for the conversational agent
and serves the assistant-ui frontend for local execution.
"""

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

logger = logging.getLogger(__name__)

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

    # Create a new agent instance for this session
    agent = ConversationalAgent()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                user_message = message_data.get("content", "")

                # Process the message with the agent
                logger.debug(f"Processing message: {user_message[:100]}...")
                response = agent.agent.run(user_message, reset=False)

                # Send response back to client
                await websocket.send_json({"role": "assistant", "content": response, "id": str(uuid.uuid4())})

            except json.JSONDecodeError:
                # Handle plain text messages for compatibility
                response = agent.agent.run(data, reset=False)
                await websocket.send_json({"role": "assistant", "content": response, "id": str(uuid.uuid4())})

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json(
                    {
                        "role": "assistant",
                        "content": f"I encountered an error: {str(e)}. Please try again.",
                        "id": str(uuid.uuid4()),
                        "error": True,
                    }
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.close()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "plexe-assistant"}
