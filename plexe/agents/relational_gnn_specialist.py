"""
Relational GNN Specialist Agent.
"""

import logging
from typing import Optional, Callable, Any

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.gnn_processing import (
    get_hetero_graph_from_registry,
    load_training_data,
    configure_temporal_sampler,
    build_gnn_model,
    train_gnn_epoch,
    evaluate_gnn,
    save_gnn_model,
    load_gnn_model,
)

logger = logging.getLogger(__name__)


class RelationalGNNSpecialistAgent:
    """
    Agent for training GNNs with Message Passing mechanism on Relational Entity Graphs.
    """

    def __init__(
        self,
        model_id: str | None = None,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the Relational GNN Specialist Agent.

        Args:
            model_id: Model ID for the LLM
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callback for chain-of-thought logging
        """
        self.model_id = model_id or "gpt-4o"
        self.verbosity = 1 if verbose else 0

        self.agent = CodeAgent(
            name="RelationalGNNSpecialist",
            description=(
                "Elite engineer in Graph Neural Networks and Deep Representation Learning. "
                "Trains end-to-end GNNs on Relational Entity Graphs using Message Passing."
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[
                # Get graph from GraphArchitect via registry
                get_hetero_graph_from_registry,
                # Load training labels from TemporalSupervisor
                load_training_data,
                # Temporal-aware data loading
                configure_temporal_sampler,
                # Model construction
                build_gnn_model,
                # Training & Evaluation
                train_gnn_epoch,
                evaluate_gnn,
                # Model persistence
                save_gnn_model,
                load_gnn_model,
            ],
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml",
                override_template_name="relational_gnn_specialist_prompt_templates.yaml",
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
                "torch",
                "torch.*",
                "torch_geometric",
                "torch_geometric.*",
                "sklearn",
                "sklearn.*",
            ],
            step_callbacks=[chain_of_thought_callable],
        )

    def run(self, task_description: str) -> Any:
        """
        Run the agent to train the GNN model.
        """
        return self.agent.run(f"Train a GNN model for the following task: {task_description}")
