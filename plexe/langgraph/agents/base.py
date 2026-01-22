"""
Base agent class for LangGraph agents.

Provides common functionality for all specialized agents.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from plexe.langgraph.config import AgentConfig, get_llm_from_model_id
from plexe.langgraph.state import PipelineState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all LangGraph agents."""
    
    def __init__(
        self,
        agent_type: str,
        config: Optional[AgentConfig] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_type: Type identifier for this agent
            config: Agent configuration (uses defaults if None)
            tools: List of tools available to this agent
        """
        self.agent_type = agent_type
        self.config = config or AgentConfig.from_env()
        self.tools = tools or []
        
        model_id = self.config.get_model_for_agent(agent_type)
        self.llm = get_llm_from_model_id(model_id, self.config.temperature)
        
        self._agent = None
    
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
    
    def get_agent(self):
        """Get or create the LangGraph agent."""
        if self._agent is None:
            self._agent = create_react_agent(
                model=self.llm,
                tools=self.tools,
                state_modifier=self.system_prompt,
            )
        return self._agent
    
    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        """
        Invoke the agent with the current state.
        
        Args:
            state: Current pipeline state
        
        Returns:
            Updated state components
        """
        agent = self.get_agent()
        
        messages = self._build_messages(state)
        
        try:
            result = agent.invoke({"messages": messages})
            return self._process_result(result, state)
        except Exception as e:
            logger.error(f"Agent {self.name} failed: {e}")
            return {
                "errors": [f"{self.name} error: {str(e)}"]
            }
    
    def _build_messages(self, state: PipelineState) -> List:
        """Build message list from state."""
        messages = []
        
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
                    "content": msg.content,
                    "timestamp": None,
                })
        
        return {"messages": new_messages}
