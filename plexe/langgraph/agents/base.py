"""
Base agent class for LangGraph agents.

Provides common functionality for all specialized agents.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents import create_agent

from plexe.langgraph.config import AgentConfig, get_llm_from_model_id
from plexe.langgraph.state import PipelineState
from plexe.langgraph.utils import BaseEmitter, ChainOfThoughtCallback
from plexe.langgraph.mcp_manager import MCPManager

logger = logging.getLogger(__name__)

def extract_text_content(content) -> str:
    """Extract text from message content (handles string or list format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for agent events with detailed chain-of-thought."""
    
    def __init__(self, agent_name: str, emitter: Optional[BaseEmitter] = None):
        self.agent_name = agent_name
        self.emitter = emitter
        self.current_thought = ""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        if self.emitter:
            model_name = ""
            if isinstance(serialized, dict):
                model_name = serialized.get("name", serialized.get("id", ["unknown"])[-1] if isinstance(serialized.get("id"), list) else "")
            self.emitter.emit_thought(self.agent_name, f"Thinking with {model_name}..." if model_name else "Analyzing...")
    
    def on_llm_end(self, response, **kwargs):
        if self.emitter and response and response.generations:
            try:
                text = extract_text_content(response.generations[0][0].text)
                if text:
                    cleaned = text.strip()
                    if len(cleaned) > 300:
                        cleaned = cleaned[:300] + "..."
                    self.current_thought = cleaned
                    self.emitter.emit_thought(self.agent_name, f"Reasoning: {cleaned}")
            except Exception as e:
                logger.debug(f"Error extracting LLM response: {e}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        if self.emitter:
            tool_name = serialized.get("name", "tool") if isinstance(serialized, dict) else "tool"
            args = {}
            if isinstance(input_str, str):
                try:
                    import json
                    args = json.loads(input_str) if input_str.startswith("{") else {"input": input_str[:100]}
                except:
                    args = {"input": str(input_str)[:100]}
            elif isinstance(input_str, dict):
                args = {k: str(v)[:50] for k, v in list(input_str.items())[:3]}
            self.emitter.emit_tool_call(self.agent_name, tool_name, args)
    
    def on_tool_end(self, output, **kwargs):
        if self.emitter and output:
            output_str = str(output) if output else ""
            if output_str:
                # Format with newlines for better readability
                formatted_output = output_str.replace('\\n', '\n')
                self.emitter.emit_thought(self.agent_name, f"Tool result:\n{formatted_output}")
    
    def on_chain_error(self, error, **kwargs):
        if self.emitter:
            self.emitter.emit_thought(self.agent_name, f"Error encountered: {str(error)[:200]}")


class BaseAgent(ABC):
    """Base class for all LangGraph agents."""
    
    def __init__(
        self,
        agent_type: str,
        config: Optional[AgentConfig] = None,
        tools: Optional[List[BaseTool]] = None,
        emitter: Optional[BaseEmitter] = None,
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_type: Type identifier for this agent
            config: Agent configuration (uses defaults if None)
            tools: List of tools available to this agent
            emitter: Optional emitter for progress callbacks
        """
        self.agent_type = agent_type
        self.config = config or AgentConfig.from_env()
        self.emitter = emitter
        
        # Initialize MCP Manager and load tools
        import asyncio
        self.mcp_manager = MCPManager()
        try:
            # We run this in a new loop or the current one to load tools during init
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.mcp_manager.initialize())
            mcp_tools = self.mcp_manager.get_tools()
            if mcp_tools:
                logger.info(f"Agent {self.name} loaded {len(mcp_tools)} MCP tools")
                self.tools.extend(mcp_tools)
        except Exception as e:
            logger.warning(f"Could not load MCP tools for {self.name}: {e}")
            
        model_id = self.config.get_model_for_agent(agent_type)
        self.llm = get_llm_from_model_id(model_id, self.config.temperature)
        
        self._agent = None
        self._callback_handler = AgentCallbackHandler(self.name, emitter)
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass
    
    @property
    def name(self) -> str:
        """Return the agent name."""
        return self.__class__.__name__
    
    @property
    def description(self) -> str:
        """Return the agent description."""
        return self.system_prompt[:200] + "..."
    
    def set_emitter(self, emitter: BaseEmitter):
        """Set the emitter for progress callbacks."""
        self.emitter = emitter
        self._callback_handler = AgentCallbackHandler(self.name, emitter)
    
    def get_agent(self):
        """Get or create the LangGraph agent."""
        if self._agent is None:
            self._agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=self.system_prompt,
            )
        return self._agent
    
    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        """
        Invoke the agent with the current state, streaming thoughts to emitter.
        
        Args:
            state: Current pipeline state
        
        Returns:
            Updated state components
        """
        agent = self.get_agent()
        
        messages = self._build_messages(state)
        logger.info(f"Agent {self.name} invoking with {len(messages)} messages")
        
        if self.emitter:
            self.emitter.emit_agent_start(self.name)
        
        try:
            config = {"callbacks": [self._callback_handler]} if self.emitter else {}
            
            result = None
            for chunk in agent.stream({"messages": messages}, config=config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if self.emitter and node_name == "agent":
                        agent_messages = node_output.get("messages", [])
                        for msg in agent_messages:
                            if isinstance(msg, AIMessage):
                                content = extract_text_content(msg.content)
                                if content and len(content) > 20:
                                    self.emitter.emit_thought(self.name, f"Thinking: {content[:400]}")
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        tool_name = tc.get("name", "unknown")
                                        tool_args = tc.get("args", {})
                                        self.emitter.emit_tool_call(self.name, tool_name, tool_args)
                result = node_output
            
            if result is None:
                result = agent.invoke({"messages": messages}, config=config)
            else:
                result = result if result.get("messages") else {"messages": []}
            
            logger.info(f"Agent {self.name} received {len(result.get('messages', []))} response messages")
            
            processed = self._process_result(result, state)
            
            if self.emitter:
                response_text = ""
                for msg in processed.get("messages", []):
                    if msg.get("role") == "assistant":
                        response_text = msg.get("content", "")[:200]
                        break
                self.emitter.emit_agent_end(self.name, response_text)
            
            return processed
        except Exception as e:
            logger.error(f"Agent {self.name} failed: {e}", exc_info=True)
            if self.emitter:
                self.emitter.emit_agent_end(self.name, f"Error: {str(e)}")
            return {
                "errors": [f"{self.name} error: {str(e)}"]
            }
    
    def _build_messages(self, state: PipelineState) -> List:
        """Build message list from state."""
        messages = [SystemMessage(content=self.system_prompt)]
        
        for msg in state.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        
        context = self._build_context(state)
        if context:
            messages.append(HumanMessage(content=f"Current context:\n{context}"))
        
        return messages
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context string from state for the agent."""
        context_parts = []
        
        if state.get("working_dir"):
            context_parts.append(f"Working directory: {state['working_dir']}")
        
        if state.get("db_connection_string"):
            context_parts.append(f"Database connection: {state['db_connection_string']}")
        
        if state.get("csv_dir"):
            context_parts.append(f"CSV directory: {state['csv_dir']}")
        
        if state.get("schema_info"):
            tables = list(state["schema_info"].get("tables", {}).keys())
            context_parts.append(f"Available tables: {', '.join(tables)}")
        
        if state.get("dataset_info"):
            context_parts.append(f"Dataset class: {state['dataset_info'].get('class_name')}")
            context_parts.append(f"Dataset file: {state['dataset_info'].get('file_path')}")
        
        if state.get("task_info"):
            context_parts.append(f"Task class: {state['task_info'].get('class_name')}")
            context_parts.append(f"Task type: {state['task_info'].get('task_type')}")
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """
        Process agent result and extract state updates.
        
        Override in subclasses to handle specific outputs.
        """
        messages = result.get("messages", [])
        
        new_messages = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                new_messages.append({
                    "role": "assistant",
                    "content": extract_text_content(msg.content),
                    "timestamp": None,
                })
        
        return {"messages": new_messages}
