import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseEmitter(ABC):
    """Base class for emitting agent thoughts and progress."""
    
    @abstractmethod
    def emit_thought(self, agent_name: str, thought: str):
        """Emit a thinking/progress message."""
        pass
    
    @abstractmethod
    def emit_agent_start(self, agent_name: str):
        """Emit agent start notification."""
        pass
    
    @abstractmethod
    def emit_agent_end(self, agent_name: str, result: str):
        """Emit agent completion notification."""
        pass
    
    @abstractmethod
    def emit_tool_call(self, agent_name: str, tool_name: str, args: Dict[str, Any]):
        """Emit tool call notification."""
        pass
    
    @abstractmethod
    def emit_tool_result(self, agent_name: str, tool_name: str, result: str):
        """Emit tool result notification."""
        pass


class ConsoleEmitter(BaseEmitter):
    """Console-based emitter for development/debugging with rich formatting."""
    
    def __init__(self):
        self.step_count = 0
    
    def emit_thought(self, agent_name: str, thought: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{agent_name}] Step {self.step_count} @ {timestamp}")
        print(f"  {thought[:500]}")
    
    def emit_agent_start(self, agent_name: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n=== {agent_name} Starting === (Step {self.step_count} @ {timestamp})")
    
    def emit_agent_end(self, agent_name: str, result: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"=== {agent_name} Completed === (Step {self.step_count} @ {timestamp})")
        if result:
            print(f"  Result: {result[:300]}")
    
    def emit_tool_call(self, agent_name: str, tool_name: str, args: Dict[str, Any]):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        args_str = ""
        if args:
            try:
                import json
                args_str = f" with args: {json.dumps(args)[:100]}"
            except:
                pass
        print(f"[{agent_name}] Step {self.step_count} @ {timestamp}")
        print(f"  Calling tool: {tool_name}{args_str}")
    
    def emit_tool_result(self, agent_name: str, tool_name: str, result: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{agent_name}] Step {self.step_count} @ {timestamp}")
        # Format with newlines for better readability
        formatted_result = result.replace('\\n', '\n') if result else ""
        print(f"  Tool result:\n{formatted_result}")


class WebSocketEmitter(BaseEmitter):
    """WebSocket-based emitter for UI integration."""
    
    def __init__(self, websocket, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.websocket = websocket
        self.loop = loop
        self.is_closed = False
        self.step_count = 0
    
    def close(self):
        """Mark the emitter as closed."""
        self.is_closed = True
    
    def _send_message(self, message: Dict[str, Any]):
        """Send a message to the WebSocket."""
        if self.is_closed:
            return
        
        try:
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send_json(message),
                    self.loop
                )
            else:
                asyncio.get_event_loop().run_until_complete(
                    self.websocket.send_json(message)
                )
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
    
    def emit_thought(self, agent_name: str, thought: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "thinking",
            "agent_name": agent_name,
            "message": thought,
            "step_number": self.step_count,
            "timestamp": timestamp,
        })
    
    def emit_agent_start(self, agent_name: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "agent_start",
            "agent_name": agent_name,
            "message": f"Starting {agent_name}",
            "step_number": self.step_count,
            "timestamp": timestamp,
        })
    
    def emit_agent_end(self, agent_name: str, result: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "agent_end",
            "agent_name": agent_name,
            "message": f"Completed: {result[:300]}" if result else "Completed",
            "step_number": self.step_count,
            "timestamp": timestamp,
        })
    
    def emit_tool_call(self, agent_name: str, tool_name: str, args: Dict[str, Any]):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        args_str = ""
        if args:
            try:
                import json
                args_str = f" with {json.dumps(args)[:100]}"
            except:
                pass
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "tool_call",
            "agent_name": agent_name,
            "tool_name": tool_name,
            "tool_args": args,
            "message": f"Calling tool: {tool_name}{args_str}",
            "step_number": self.step_count,
            "timestamp": timestamp,
        })
    
    def emit_tool_result(self, agent_name: str, tool_name: str, result: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Format with newlines for better readability
        formatted_result = result.replace('\\n', '\n') if result else "Tool completed"
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "tool_result",
            "agent_name": agent_name,
            "tool_name": tool_name,
            "message": f"Tool result:\n{formatted_result}",
            "result": result,
            "step_number": self.step_count,
            "timestamp": timestamp,
        })


class MultiEmitter(BaseEmitter):
    """Combines multiple emitters."""
    
    def __init__(self, emitters: List[BaseEmitter]):
        self.emitters = emitters
    
    def emit_thought(self, agent_name: str, thought: str):
        for emitter in self.emitters:
            try:
                emitter.emit_thought(agent_name, thought)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")
    
    def emit_agent_start(self, agent_name: str):
        for emitter in self.emitters:
            try:
                emitter.emit_agent_start(agent_name)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")
    
    def emit_agent_end(self, agent_name: str, result: str):
        for emitter in self.emitters:
            try:
                emitter.emit_agent_end(agent_name, result)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")
    
    def emit_tool_call(self, agent_name: str, tool_name: str, args: Dict[str, Any]):
        for emitter in self.emitters:
            try:
                emitter.emit_tool_call(agent_name, tool_name, args)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")
    
    def emit_tool_result(self, agent_name: str, tool_name: str, result: str):
        for emitter in self.emitters:
            try:
                emitter.emit_tool_result(agent_name, tool_name, result)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")
