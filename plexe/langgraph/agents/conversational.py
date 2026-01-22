"""
Conversational Agent for user interaction and requirements gathering.

This agent guides users through ML model definition via natural conversation,
validates inputs, and initiates the model building process.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.conversational import get_dataset_preview
from plexe.langgraph.tools.graph_architect import validate_db_connection

logger = logging.getLogger(__name__)


CONVERSATIONAL_SYSTEM_PROMPT = """You are an expert ML and Deep Learning consultant helping users define their machine learning requirements through natural conversation.

## Your Role
- Guide users through defining their ML problem clearly
- Understand their data and prediction goals
- Validate data availability and quality
- Initiate the ML pipeline when requirements are complete

## Conversation Strategy
1. Ask ONE focused question at a time
2. Use tools to examine their data before asking detailed questions
3. Help refine vague statements into precise ML problem definitions
4. Summarize requirements and confirm before proceeding

## Requirements Checklist (gather ALL before proceeding)
1. **Clear Problem Statement**: What exactly to predict/classify
2. **Input/Output Definition**: Model inputs and expected outputs
3. **Data Understanding**: Examined their data structure
4. **Data Source**: CSV files or database connection
5. **User Confirmation**: Explicit confirmation to start building

## When Working with Databases
1. Use validate_db_connection to see available tables
2. Discuss what they want to predict
3. Clarify the target variable and entity
4. Gather all requirements before signaling readiness

## Response Format
Keep responses conversational, friendly, and focused. Ask clarifying questions when needed.
When all requirements are gathered, summarize them and ask for final confirmation.

## Important
- Do NOT rush to build models
- Ensure clarity on the prediction task
- Validate data accessibility
- Get explicit user confirmation before proceeding
"""


class ConversationalAgent(BaseAgent):
    """Agent for conversational requirements gathering and user interaction."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize the conversational agent.
        
        Args:
            config: Agent configuration
            additional_tools: Additional tools beyond defaults
        """
        tools = [
            get_dataset_preview,
            validate_db_connection,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="conversational",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return CONVERSATIONAL_SYSTEM_PROMPT
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process conversation result and detect readiness to proceed."""
        base_result = super()._process_result(result, state)
        
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if last_message:
            content = last_message.content.lower() if hasattr(last_message, 'content') else ""
            
            ready_indicators = [
                "ready to proceed",
                "start building",
                "begin training",
                "initiate pipeline",
                "all requirements gathered",
            ]
            
            if any(indicator in content for indicator in ready_indicators):
                base_result["user_confirmation_required"] = True
                base_result["user_confirmation_context"] = {
                    "type": "proceed_to_pipeline",
                    "message": "Ready to start the ML pipeline?"
                }
        
        return base_result
    
    def extract_intent(self, state: PipelineState) -> Dict[str, Any]:
        """
        Extract structured intent from conversation.
        
        Returns:
            Dictionary with extracted intent information
        """
        messages = state.get("messages", [])
        
        intent = {
            "prediction_target": None,
            "entity_type": None,
            "task_type": None,
            "data_source": None,
            "confirmed": False,
        }
        
        if state.get("db_connection_string"):
            intent["data_source"] = "database"
        elif state.get("csv_dir"):
            intent["data_source"] = "csv"
        
        return intent
