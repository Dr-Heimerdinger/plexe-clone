"""
System prompts for LangGraph agents.

Each agent has its own prompt file defining its role, workflow, and guidelines.
"""

from plexe.langgraph.prompts.conversational import CONVERSATIONAL_SYSTEM_PROMPT
from plexe.langgraph.prompts.eda import EDA_SYSTEM_PROMPT
from plexe.langgraph.prompts.dataset_builder import DATASET_BUILDER_SYSTEM_PROMPT
from plexe.langgraph.prompts.task_builder import TASK_BUILDER_SYSTEM_PROMPT
from plexe.langgraph.prompts.gnn_specialist import GNN_SPECIALIST_SYSTEM_PROMPT
from plexe.langgraph.prompts.operation import OPERATION_SYSTEM_PROMPT

__all__ = [
    "CONVERSATIONAL_SYSTEM_PROMPT",
    "EDA_SYSTEM_PROMPT",
    "DATASET_BUILDER_SYSTEM_PROMPT",
    "TASK_BUILDER_SYSTEM_PROMPT",
    "GNN_SPECIALIST_SYSTEM_PROMPT",
    "OPERATION_SYSTEM_PROMPT",
]
