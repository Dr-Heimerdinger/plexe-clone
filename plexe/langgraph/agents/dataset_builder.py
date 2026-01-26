"""
Dataset Builder Agent.

This agent builds RelBench Database objects from CSV files,
generating complete Python Dataset classes for GNN training.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.common import save_artifact
from plexe.langgraph.tools.dataset_builder import (
    get_csv_files_info,
    get_temporal_statistics,
    register_dataset_code,
)
from plexe.langgraph.prompts.dataset_builder import DATASET_BUILDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class DatasetBuilderAgent(BaseAgent):
    """Agent for building RelBench Dataset classes from CSV data."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        tools = [
            get_csv_files_info,
            get_temporal_statistics,
            register_dataset_code,
            save_artifact,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="dataset_builder",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return DATASET_BUILDER_SYSTEM_PROMPT
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with CSV, schema, and EDA information."""
        context_parts = []
        
        if state.get("working_dir"):
            context_parts.append(f"Working directory: {state['working_dir']}")
        
        if state.get("csv_dir"):
            context_parts.append(f"CSV directory: {state['csv_dir']}")
        
        if state.get("schema_info"):
            schema = state["schema_info"]
            tables = list(schema.get("tables", {}).keys())
            context_parts.append(f"Tables: {', '.join(tables)}")
            
            if schema.get("relationships"):
                rels = []
                for r in schema["relationships"]:
                    rels.append(f"{r['source_table']}.{r['source_column']} -> {r['target_table']}")
                context_parts.append(f"Foreign keys: {'; '.join(rels)}")
            
            if schema.get("temporal_columns"):
                for table, cols in schema["temporal_columns"].items():
                    context_parts.append(f"{table} time columns: {cols}")
        
        if state.get("eda_info"):
            eda = state["eda_info"]
            context_parts.append("\n## EDA Analysis Results:")
            
            if eda.get("quality_issues"):
                context_parts.append("Data Quality Issues:")
                for table, issues in eda["quality_issues"].items():
                    if issues:
                        context_parts.append(f"  - {table}: {issues}")
            
            if eda.get("temporal_analysis"):
                context_parts.append("Temporal Analysis:")
                for table, analysis in eda["temporal_analysis"].items():
                    if analysis.get("time_columns"):
                        context_parts.append(f"  - {table}: time columns = {analysis['time_columns']}")
            
            if eda.get("suggested_splits"):
                splits = eda["suggested_splits"]
                if splits.get("val_timestamp"):
                    context_parts.append(f"Suggested val_timestamp: {splits['val_timestamp']}")
                if splits.get("test_timestamp"):
                    context_parts.append(f"Suggested test_timestamp: {splits['test_timestamp']}")
            
            if eda.get("relationship_analysis"):
                context_parts.append("Relationship Analysis:")
                for key, info in eda["relationship_analysis"].items():
                    if isinstance(info, dict) and info.get("cardinality"):
                        context_parts.append(f"  - {key}: {info['cardinality']}")
        
        working_dir = state.get('working_dir', '')
        csv_dir = state.get('csv_dir', '')
        
        task_instruction = f"""
YOUR TASK:
1. Call get_csv_files_info("{csv_dir}") to list all CSV files and columns
2. Call get_temporal_statistics("{csv_dir}") to analyze timestamps
3. Generate a complete GenDataset class following the template above
4. Call register_dataset_code(code, "GenDataset", "{working_dir}/dataset.py") to save it

IMPORTANT: You MUST call register_dataset_code to save the dataset.py file!
"""
        context_parts.append(task_instruction)
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and extract dataset information."""
        base_result = super()._process_result(result, state)
        
        dataset_info = {}
        working_dir = state.get("working_dir", "")
        
        # Check if dataset.py was created
        dataset_path = os.path.join(working_dir, "dataset.py")
        if os.path.exists(dataset_path):
            dataset_info["class_name"] = "GenDataset"
            dataset_info["file_path"] = dataset_path
            logger.info(f"Dataset file created at: {dataset_path}")
        else:
            logger.warning(f"Dataset file not found at: {dataset_path}")
            dataset_info["class_name"] = "GenDataset"
            dataset_info["file_path"] = dataset_path
        
        base_result["dataset_info"] = dataset_info
        base_result["current_phase"] = PipelinePhase.TASK_BUILDING.value
        
        return base_result
