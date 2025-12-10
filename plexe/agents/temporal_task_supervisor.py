"""
Temporal Task Supervisor Agent.

This agent is responsible for creating temporal train/val/test datasets
following Relational Deep Learning (RDL) principles.

SIMPLIFIED DESIGN:
- Primary tool: create_temporal_dataset() - does everything needed
- Helper tools: discover_temporal_columns(), get_table_columns() - for exploration
- Validation tool: validate_temporal_consistency() - for post-graph validation

REMOVED TOOLS (redundant/confusing):
- execute_sql_query() - low-level, error-prone for agents
- define_training_task() - metadata only, merged into create_temporal_dataset
- generate_sql_implementation() - SQL template only, merged into create_temporal_dataset
- temporal_split() - validation merged into create_temporal_dataset
- generate_temporal_splits_from_db() - misleading name, doesn't create datasets
"""

import logging
from typing import Optional, Callable, Any

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.temporal_processing import (
    discover_temporal_columns,
    get_table_columns,
    create_temporal_dataset,
    validate_temporal_consistency,
)

logger = logging.getLogger(__name__)


class TemporalTaskSupervisorAgent:
    """
    Agent for creating temporal train/val/test datasets in RDL pipeline.
    
    This agent ensures:
    1. Temporal consistency (no data leakage)
    2. Proper train/val/test splits with sliding window sampling
    3. Registration of datasets to ObjectRegistry for subsequent agents
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
                "Creates temporal train/val/test datasets for Relational Deep Learning. "
                "Ensures temporal consistency and prevents data leakage. "
                "IMPORTANT: Database columns use snake_case (e.g., owner_user_id, NOT OwnerUserId)."
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[
                # Core exploration tools
                discover_temporal_columns,  # Discover schema and date ranges
                get_table_columns,          # Verify column names before writing SQL
                
                # Primary dataset creation tool (MUST USE)
                create_temporal_dataset,    # Creates and registers train/val/test datasets
                
                # Post-graph validation (optional, used after GraphArchitect)
                validate_temporal_consistency,
            ],
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml",
                override_template_name="temporal_task_supervisor_prompt_templates.yaml",
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
            ],
            step_callbacks=[chain_of_thought_callable],
        )

    def run(self, task_description: str) -> Any:
        """
        Run the agent to define the task and split the data.
        
        This method ensures that the agent not only plans the temporal splits
        but also CREATES the actual train/val/test datasets required by
        subsequent agents (RelationalGraphArchitect, RelationalGNNSpecialist).
        """
        result = self.agent.run(f"Define the temporal task and split strategy for: {task_description}")
        
        # Validate that required datasets were created
        self._validate_datasets_created()
        
        return result
    
    def _validate_datasets_created(self) -> None:
        """
        Validates that the agent successfully created the required datasets.
        
        Raises:
            RuntimeError: If any required dataset is missing from ObjectRegistry
        """
        from plexe.core.object_registry import ObjectRegistry
        from plexe.internal.common.datasets.interface import TabularConvertible
        
        registry = ObjectRegistry()
        required_datasets = ["temporal_train", "temporal_val", "temporal_test"]
        missing_datasets = []
        
        for dataset_name in required_datasets:
            try:
                dataset = registry.get(TabularConvertible, dataset_name)
                if dataset is None:
                    missing_datasets.append(dataset_name)
                else:
                    logger.info(f"✅ Validated dataset: {dataset_name} ({len(dataset.to_pandas())} samples)")
            except KeyError:
                missing_datasets.append(dataset_name)
        
        if missing_datasets:
            error_msg = (
                f"❌ TemporalTaskSupervisor FAILED to create required datasets!\n\n"
                f"Missing datasets: {missing_datasets}\n\n"
                f"CAUSE: The agent did not call create_temporal_dataset().\n\n"
                f"IMPACT: Subsequent agents (RelationalGraphArchitect, RelationalGNNSpecialist) will CRASH!\n\n"
                f"FIX: The agent MUST call create_temporal_dataset() with:\n"
                f"  - db_connection_string\n"
                f"  - entity_table, entity_id_column, timestamp_column\n"
                f"  - label_query (SQL with {{start_date}}, {{end_date}} placeholders)\n"
                f"  - feature_query (SQL with {{cutoff_date}} placeholder)\n"
                f"  - train_end_date, val_end_date\n"
                f"  - window_size_days, num_train_windows, train_stride_days\n\n"
                f"This will register 'temporal_train', 'temporal_val', 'temporal_test' to ObjectRegistry.\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("✅ All required temporal datasets have been created and validated")

