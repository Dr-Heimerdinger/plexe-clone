"""
Dataset Builder Agent for constructing RelBench Database objects from raw CSV files.

This agent builds a plexe.relbench.base.Database by:
1. Reading CSV files from a specified directory
2. Using EDA reports to determine data processing steps
3. Processing time columns (merging date/time, propagating timestamps)
4. Defining schema with primary keys, foreign keys, and time columns
5. Returning a complete Database object
"""

import logging
from typing import Optional, Callable

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.datasets import get_dataset_reports, get_latest_datasets
from plexe.tools.database_builder import (
    get_csv_files_from_path,
    get_schema_metadata_from_registry,
    register_database_code,
    export_database_code,
    validate_database_code,
    get_temporal_statistics,
)

logger = logging.getLogger(__name__)


class DatasetBuilderAgent:
    """
    Agent for building RelBench Database objects from raw CSV data.
    
    This agent analyzes CSV files, applies data processing based on EDA reports,
    handles temporal data properly, and constructs a Database with correct
    schema definitions (primary keys, foreign keys, time columns).
    """

    def __init__(
        self,
        model_id: str | None = None,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the Dataset Builder Agent.

        Args:
            model_id: Model ID for the LLM
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callback for chain-of-thought logging
        """
        self.model_id = model_id or "gemini/gemini-2.5-flash"
        self.verbosity = 1 if verbose else 0

        self.agent = CodeAgent(
            name="DatasetBuilder",
            description=(
                "Expert in building RelBench Database objects from raw CSV files. "
                "Reads CSV files, applies data cleaning based on EDA reports, "
                "processes temporal columns (merging date/time, propagating timestamps from parent to child tables), "
                "defines schema with primary keys, foreign keys, and time columns, "
                "and generates a complete Python Dataset class that can be registered and exported."
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[
                # Data discovery
                get_csv_files_from_path,
                get_schema_metadata_from_registry,
                get_dataset_reports,
                get_latest_datasets,
                # Temporal analysis
                get_temporal_statistics,
                # Database building
                register_database_code,
                export_database_code,
                validate_database_code,
            ],
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml",
                override_template_name="dataset_builder_prompt_templates.yaml",
            ),
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
                "os",
                "os.*",
                "pathlib",
                "pathlib.*",
                "plexe.relbench",
                "plexe.relbench.*",
            ],
            step_callbacks=[chain_of_thought_callable],
            max_steps=30,
        )

    def run(self, task_description: str) -> str:
        """
        Run the agent to build a Database from CSV files.
        
        Args:
            task_description: Description of the database building task
            
        Returns:
            Result of the database building process
        """
        return self.agent.run(task_description)
