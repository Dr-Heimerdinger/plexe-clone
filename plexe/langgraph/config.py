"""
Configuration for LangGraph-based agents.

This module provides configuration management for agent models
using environment variables and defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for agent models from environment variables."""
    
    orchestrator_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_ORCHESTRATOR_MODEL", "openai/gpt-4o"
    ))
    conversational_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_CONVERSATIONAL_MODEL", "openai/gpt-4o"
    ))
    eda_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_EDA_MODEL", "openai/gpt-4o"
    ))
    dataset_builder_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_DATASET_BUILDER_MODEL", "openai/gpt-4o"
    ))
    task_builder_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_TASK_BUILDER_MODEL", "openai/gpt-4o"
    ))
    gnn_specialist_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_GNN_SPECIALIST_MODEL", "openai/gpt-4o"
    ))
    operation_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_OPERATION_MODEL", "openai/gpt-4o"
    ))
    
    temperature: float = field(default_factory=lambda: float(os.environ.get(
        "PLEXE_AGENT_TEMPERATURE", "0.1"
    )))
    max_retries: int = field(default_factory=lambda: int(os.environ.get(
        "PLEXE_MAX_RETRIES", "3"
    )))
    verbose: bool = field(default_factory=lambda: os.environ.get(
        "PLEXE_VERBOSE", "false"
    ).lower() == "true")
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables."""
        return cls()
    
    def get_model_for_agent(self, agent_type: str) -> str:
        """Get the model ID for a specific agent type."""
        mapping = {
            "orchestrator": self.orchestrator_model,
            "conversational": self.conversational_model,
            "eda": self.eda_model,
            "dataset_builder": self.dataset_builder_model,
            "task_builder": self.task_builder_model,
            "gnn_specialist": self.gnn_specialist_model,
            "operation": self.operation_model,
        }
        return mapping.get(agent_type, self.orchestrator_model)


def get_llm_from_model_id(model_id: str, temperature: float = 0.1):
    """
    Create a LangChain LLM instance from a model ID string.
    
    Supports formats:
    - openai/gpt-4o -> OpenAI
    - anthropic/claude-sonnet-4-20250514 -> Anthropic
    - gemini/gemini-2.5-flash -> Google Gemini
    """
    from langchain_core.language_models import BaseChatModel
    
    if model_id.startswith("openai/"):
        from langchain_openai import ChatOpenAI
        model_name = model_id.replace("openai/", "")
        return ChatOpenAI(model=model_name, temperature=temperature)
    
    elif model_id.startswith("anthropic/"):
        from langchain_anthropic import ChatAnthropic
        model_name = model_id.replace("anthropic/", "")
        return ChatAnthropic(model=model_name, temperature=temperature)
    
    elif model_id.startswith("gemini/"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = model_id.replace("gemini/", "")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, temperature=temperature)
