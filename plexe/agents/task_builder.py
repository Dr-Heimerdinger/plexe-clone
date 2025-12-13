"""
Task Builder Agent for constructing RelBench Task objects from Database/Dataset.

This agent builds EntityTask or RecommendationTask by:
1. Analyzing the database schema to understand available tables and relationships
2. Identifying entity tables and potential prediction targets
3. Defining the prediction task (classification, regression, or recommendation)
4. Generating SQL queries to compute labels from historical data
5. Returning a complete Task class that can be used for training

The Task is the "question" that we want the AI model to answer, while
the Dataset is the "raw material" used to answer that question.
"""

import logging
from typing import Optional, Callable

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.datasets import get_dataset_reports, get_latest_datasets
from plexe.tools.task_builder import (
    get_database_schema_for_task,
    get_dataset_info,
    get_available_metrics,
    get_table_sample_data,
    test_sql_query,
    analyze_potential_tasks,
    validate_task_code,
    register_task_code,
    export_task_code,
    get_entity_task_template,
    get_recommendation_task_template,
    get_sql_query_examples,
)

logger = logging.getLogger(__name__)


class TaskBuilderAgent:
    """
    Agent for building RelBench Task objects from Database/Dataset.
    
    This agent analyzes the database schema, understands entity relationships,
    and constructs prediction tasks (EntityTask or RecommendationTask) with
    proper SQL queries to compute labels from historical data.
    
    Tasks define what prediction we want to make:
    - EntityTask: Classification or regression over a single entity
      (e.g., predict if a user will churn, predict driver DNF)
    - RecommendationTask: Link prediction between entities
      (e.g., predict which products a user will buy)
    """

    def __init__(
        self,
        model_id: str | None = None,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the Task Builder Agent.

        Args:
            model_id: Model ID for the LLM
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callback for chain-of-thought logging
        """
        self.model_id = model_id or "gemini/gemini-2.5-flash"
        self.verbosity = 1 if verbose else 0

        self.agent = CodeAgent(
            name="TaskBuilder",
            description=(
                "Expert in building RelBench Task objects (EntityTask or RecommendationTask) "
                "from a Database/Dataset. Analyzes database schema to understand entity relationships, "
                "defines prediction tasks based on user requirements, generates efficient SQL queries "
                "using DuckDB to compute labels from historical data, and produces complete Python "
                "Task classes that can be registered and used for GNN training."
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[
                # Schema and data discovery
                get_database_schema_for_task,
                get_dataset_info,
                get_table_sample_data,
                get_dataset_reports,
                get_latest_datasets,
                # Task analysis
                analyze_potential_tasks,
                get_available_metrics,
                # SQL query testing
                test_sql_query,
                # Code generation support
                get_entity_task_template,
                get_recommendation_task_template,
                get_sql_query_examples,
                # Validation and registration
                validate_task_code,
                register_task_code,
                export_task_code,
            ],
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml",
                override_template_name="task_builder_prompt_templates.yaml",
            ),
            verbosity_level=self.verbosity,
            add_base_tools=False,
            additional_authorized_imports=config.code_generation.authorized_agent_imports
            + [
                "plexe",
                "plexe.*",
                "plexe.relbench",
                "plexe.relbench.*",
                "plexe.relbench.base",
                "plexe.relbench.metrics",
                "pandas",
                "pandas.*",
                "numpy",
                "numpy.*",
                "duckdb",
                "duckdb.*",
                "os",
                "os.*",
                "pathlib",
                "pathlib.*",
            ],
            step_callbacks=[chain_of_thought_callable],
            max_steps=30,
        )

    def run(self, task_description: str) -> str:
        """
        Run the agent to build a Task from the Database.
        
        Args:
            task_description: Description of the prediction task to build.
                Should include:
                - What entity to predict for (e.g., users, drivers)
                - What to predict (e.g., churn, count of events, recommendations)
                - Time window for prediction (optional, defaults to 30 days)
                - Task type if known (classification, regression, recommendation)
            
        Returns:
            Result of the task building process including generated code
        """
        return self.agent.run(task_description)
