"""
Feature Generation Agent for combining datasets using featuretools.
"""

import logging
import os
from typing import Optional, Callable

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.core.object_registry import ObjectRegistry
from plexe.tools.datasets import (
    get_dataset_preview,
    get_latest_datasets,
    register_feature_engineering_report,
)
from plexe.tools.execution import execute_code
from plexe.tools.io_manager import read_file, write_file

logger = logging.getLogger(__name__)


class FeatureGeneratorAgent:
    """
    Agent for combining datasets using featuretools.
    """

    def __init__(
        self,
        session_id: str,
        tables: list,
        relationships: list,
        model_id: str | None = None,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the feature generator agent.

        Args:
            model_id: Model ID for the LLM to use for feature generation
            session_id: The session ID for the current run
            tables: A list of tables to be combined
            relationships: A list of relationships between the tables
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callback for chain-of-thought logging
        """
        env_model = os.environ.get("PLEXE_CONVERSATIONAL_MODEL")

        if model_id:
            self.model_id = model_id
        elif env_model:
            self.model_id = env_model
        else:
            # Auto-detect from available API keys
            if os.environ.get("GEMINI_API_KEY"):
                self.model_id = "gemini/gemini-2.5-flash"
            elif os.environ.get("OPENAI_API_KEY"):
                self.model_id = "openai/gpt-4o-mini"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self.model_id = "anthropic/claude-sonnet-4-20250514"
            else:
                self.model_id = "anthropic/claude-sonnet-4-20250514"
        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Register session_id for tools to use
        ObjectRegistry().register(str, "session_id", session_id, overwrite=True)

        # Create feature generator agent
        self.agent = CodeAgent(
            name="FeatureGenerator",
            description=(
                "Expert data scientist that combines datasets using featuretools. "
                "To work effectively, as part of the 'task' prompt the agent STRICTLY requires:"
                "- the list of tables"
                "- the list of relationships"
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[
                get_dataset_preview,
                get_latest_datasets,
                register_feature_engineering_report,
                execute_code,
                read_file,
                write_file,
            ],
            add_base_tools=False,
            additional_authorized_imports=config.code_generation.authorized_agent_imports
            + [
                "featuretools",
                "featuretools.*",
                "pandas",
                "pandas.*",
                "numpy",
                "numpy.*",
            ],
            verbosity_level=self.verbosity,
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml",
                override_template_name="feature_generator_prompt_templates.yaml",
                template_vars={"tables": tables, "relationships": relationships},
            ),
            planning_interval=5,
            step_callbacks=[chain_of_thought_callable],
        )

    def run(self, task: str):
        """
        Run the agent to perform the feature generation task.

        Args:
            task: The task to be performed by the agent.
        """
        return self.agent.run(task)
