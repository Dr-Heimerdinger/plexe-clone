"""
Temporal Task Supervisor Agent.
"""

import logging
from typing import Optional, Callable, Any

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.temporal_processing import (
    generate_training_table_sql,
    temporal_split,
    validate_temporal_consistency,
)

logger = logging.getLogger(__name__)


class TemporalTaskSupervisorAgent:
    """
    Agent for defining predictive tasks and enforcing temporal consistency in RDL.
    """

    def __init__(
        self,
        model_id: str | None = None,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the Temporal Task Supervisor Agent.

        Args:
            model_id: Model ID for the LLM
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callback for chain-of-thought logging
        """
        self.model_id = model_id or "gpt-4o"
        self.verbosity = 1 if verbose else 0

        self.agent = CodeAgent(
            name="TemporalTaskSupervisor",
            description=(
                "Guardian of temporal integrity and causality in the RDL pipeline. "
                "Constructs Training Tables and enforces Time-Consistent Splitting."
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[
                generate_training_table_sql,
                temporal_split,
                validate_temporal_consistency,
            ],
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml",
                override_template_name="temporal_task_supervisor_prompt_templates.yaml",
            ),
            # verbose=verbose, # Removed verbose argument
            verbosity_level=self.verbosity,
            add_base_tools=False,
            additional_authorized_imports=config.code_generation.authorized_agent_imports
            + [
                "plexe",
                "plexe.*",
                "pandas",
                "pandas.*",
                "numpy",
                "numpy.*",
            ],
            step_callbacks=[chain_of_thought_callable],
        )

    def run(self, task_description: str) -> Any:
        """
        Run the agent to define the task and split the data.
        """
        return self.agent.run(f"Define the temporal task and split strategy for: {task_description}")
