"""
FastAPI server for the Plexe LangGraph-based multi-agent system.

This module provides WebSocket API for real-time chat communication
and serves the frontend for the Plexe ML platform.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from plexe.langgraph import PlexeOrchestrator, AgentConfig
from plexe.api import datasets_router

logger = logging.getLogger(__name__)

app = FastAPI(title="Plexe Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)

ui_dir = Path(__file__).parent / "ui"
frontend_dist = ui_dir / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dist)), name="static")
elif ui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")


class SessionManager:
    """Manages active chat sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, session_id: str) -> PlexeOrchestrator:
        """Create a new session with its own orchestrator."""
        config = AgentConfig.from_env()
        orchestrator = PlexeOrchestrator(config=config, verbose=True)
        self.sessions[session_id] = {
            "orchestrator": orchestrator,
            "working_dir": f"workdir/session-{session_id}",
        }
        return orchestrator
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get an existing session."""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


session_manager = SessionManager()


@app.get("/")
async def root():
    """Serve the main HTML page."""
    built_index = frontend_dist / "index.html"
    legacy_index = ui_dir / "index.html"

    if built_index.exists():
        return FileResponse(str(built_index))
    if legacy_index.exists():
        return FileResponse(str(legacy_index))
    return {
        "error": "Frontend not found. Please ensure plexe/ui/frontend/dist/index.html exists."
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"New WebSocket connection: {session_id}")

    loop = asyncio.get_running_loop()
    orchestrator = session_manager.create_session(session_id)
    session_data = session_manager.get_session(session_id)
    working_dir = session_data["working_dir"]
    
    agent_task = None
    is_closed = False

    async def send_message(msg_type: str, content: Any):
        """Send a message to the client."""
        if not is_closed:
            try:
                await websocket.send_json({
                    "type": msg_type,
                    "content": content,
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                })
            except Exception as e:
                logger.warning(f"Failed to send message: {e}")

    async def send_thought(agent_name: str, thought: str):
        """Send a thought/progress update to the client."""
        await send_message("thought", {
            "agent": agent_name,
            "message": thought,
        })

    async def run_agent_task(user_message: str, db_connection: Optional[str] = None):
        """Run the orchestrator in a separate thread."""
        nonlocal agent_task
        
        try:
            await send_thought("system", "Processing your request...")
            
            def run_sync():
                session = session_manager.get_session(session_id)
                if not session:
                    return {"status": "error", "error": "Session not found"}
                
                orch = session["orchestrator"]
                state = orch.get_session_state(session_id)
                
                if state:
                    return orch.chat(
                        message=user_message,
                        session_id=session_id,
                        working_dir=working_dir,
                    )
                else:
                    return orch.run(
                        user_message=user_message,
                        db_connection_string=db_connection,
                        working_dir=working_dir,
                        session_id=session_id,
                    )
            
            result = await loop.run_in_executor(None, run_sync)
            
            response_text = ""
            if result.get("status") == "success":
                response_text = result.get("response", "Processing complete.")
            elif result.get("status") == "completed":
                state = result.get("state", {})
                messages = state.get("messages", [])
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        response_text = msg.get("content", "")
                        break
                if not response_text:
                    response_text = "Pipeline completed successfully."
            elif result.get("status") == "error":
                response_text = f"Error: {result.get('error', 'Unknown error')}"
            else:
                response_text = result.get("response", str(result))
            
            if not is_closed:
                await websocket.send_json({
                    "role": "assistant",
                    "content": response_text,
                    "id": str(uuid.uuid4()),
                    "phase": result.get("phase", result.get("state", {}).get("current_phase")),
                })
                
        except asyncio.CancelledError:
            logger.info("Agent task cancelled")
            raise
        except Exception as e:
            logger.error(f"Agent error: {e}")
            if not is_closed:
                await websocket.send_json({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}. Please try again.",
                    "id": str(uuid.uuid4()),
                    "error": True,
                })
        finally:
            agent_task = None

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                
                if message_data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if message_data.get("type") == "stop":
                    logger.info("Stop command received")
                    if agent_task:
                        agent_task.cancel()
                        try:
                            await agent_task
                        except asyncio.CancelledError:
                            pass
                        agent_task = None
                    continue

                if message_data.get("type") == "confirmation_response":
                    confirmed = message_data.get("confirmed", False)
                    logger.info(f"Received confirmation: {confirmed}")
                    continue
                
                user_message = message_data.get("content", "")
                db_connection = message_data.get("db_connection_string")
                
                if not user_message:
                    continue

                logger.debug(f"Processing message: {user_message[:100]}...")
                
                if agent_task is None:
                    agent_task = asyncio.create_task(
                        run_agent_task(user_message, db_connection)
                    )
                else:
                    logger.warning("Agent is already processing")
                    await websocket.send_json({
                        "role": "assistant",
                        "content": "I'm still processing your previous request. Please wait.",
                        "id": str(uuid.uuid4()),
                    })

            except json.JSONDecodeError:
                if agent_task is None:
                    agent_task = asyncio.create_task(run_agent_task(data))

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                try:
                    await websocket.send_json({
                        "role": "assistant",
                        "content": f"I encountered an error: {str(e)}. Please try again.",
                        "id": str(uuid.uuid4()),
                        "error": True,
                    })
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        is_closed = True
        if agent_task:
            agent_task.cancel()
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        is_closed = True
        if agent_task:
            agent_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        is_closed = True
        session_manager.remove_session(session_id)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "plexe-assistant", "version": "2.0.0"}
