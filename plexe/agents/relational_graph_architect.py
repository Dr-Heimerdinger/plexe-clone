"""
Relational Graph Architect Agent.
"""

import logging
from typing import Optional, Callable, Any

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.graph_processing import (
    extract_schema_metadata,
    build_hetero_graph,
    encode_multi_modal_features,
)

logger = logging.getLogger(__name__)


class RelationalGraphArchitectAgent:
    """
    Agent for transforming relational databases into heterogeneous graphs.
    """

    def __init__(
        self,
        model_id: str | None = None,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the Relational Graph Architect Agent.

        Args:
            model_id: Model ID for the LLM
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callback for chain-of-thought logging
        """
        self.model_id = model_id or "gpt-4o"
        self.verbosity = 1 if verbose else 0

        self.agent = CodeAgent(
            name="RelationalGraphArchitect",
            description=(
                "Expert in Graph Representation Learning and Heterogeneous Graph Construction. "
                "Transforms multi-table relational databases into Heterogeneous Graphs for GNNs."
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[
                extract_schema_metadata,
                build_hetero_graph,
                encode_multi_modal_features,
            ],
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml",
                override_template_name="relational_graph_architect_prompt_templates.yaml",
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
                "sqlalchemy",
                "sqlalchemy.*",
                "psycopg2",
                "psycopg2.*",
            ],
            step_callbacks=[chain_of_thought_callable],
        )

    def run(self, db_connection: Any) -> Any:
        """
        Run the agent to build the graph.
        """
        return self.agent.run(f"Build a heterogeneous graph from the database connection: {db_connection}")
