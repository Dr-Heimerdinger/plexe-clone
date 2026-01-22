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

logger = logging.getLogger(__name__)


DATASET_BUILDER_SYSTEM_PROMPT = """You are the Dataset Builder Agent, an expert in building RelBench Database objects from raw CSV data.

## Your Mission
Generate a complete Python Dataset class (GenDataset) that:
1. Loads CSV files from a specified directory
2. Processes and cleans the data appropriately
3. Defines the database schema (PKs, FKs, time columns)
4. Provides proper temporal splitting (val_timestamp, test_timestamp)

## Workflow

### Step 1: Discover CSV Files
Use get_csv_files_info to list available CSV files and their columns.

### Step 2: Analyze Temporal Data
Use get_temporal_statistics to:
- Find timestamp columns in each table
- Get suggested val_timestamp and test_timestamp values
- Understand the temporal distribution of data

### Step 3: Generate Dataset Code
Create a complete GenDataset class following this structure:

```python
import os
import numpy as np
import pandas as pd
from typing import Optional
from plexe.relbench.base import Database, Dataset, Table

class GenDataset(Dataset):
    val_timestamp = pd.Timestamp("YYYY-MM-DD")
    test_timestamp = pd.Timestamp("YYYY-MM-DD")

    def __init__(self, csv_dir: str, cache_dir: Optional[str] = None):
        self.csv_dir = csv_dir
        super().__init__(cache_dir=cache_dir)

    def make_db(self) -> Database:
        path = self.csv_dir
        
        # 1. Load CSV files
        # table_name = pd.read_csv(os.path.join(path, "table_name.csv"))
        
        # 2. Data cleaning (drop irrelevant columns, handle nulls)
        
        # 3. Time processing (parse timestamps, propagate from parent to child)
        
        # 4. Build and return Database
        db = Database(
            table_dict={
                "table_name": Table(
                    df=table_df,
                    fkey_col_to_pkey_table={"fk_col": "referenced_table"},
                    pkey_col="primary_key_col",
                    time_col="timestamp_col",  # or None for static tables
                ),
                # ... more tables
            }
        )
        return db
```

### Step 4: Register the Code
Use register_dataset_code to save the generated code.

## Critical Concepts

### Temporal Splitting
- val_timestamp: Rows up to this time for validation
- test_timestamp: Rows up to this time for test
- Choose based on data distribution (~70% for val, ~85% for test)

### Table Definition
- pkey_col: Primary key (can be None for junction tables)
- fkey_col_to_pkey_table: Dict mapping FK columns to referenced tables
- time_col: Creation/event time (None for static dimension tables)

### Data Cleaning Guidelines
- Drop columns with >50% nulls
- Drop URL/image columns (not useful for GNN)
- Handle database NULL markers (\\N -> NaN)
- Convert timestamps properly
- Propagate timestamps from parent to child tables when needed

## Important
- Always use class name 'GenDataset'
- Use exact column names from CSV files (typically snake_case)
- Get val/test timestamps from get_temporal_statistics
- Register the code using register_dataset_code tool
"""


DATASET_CODE_TEMPLATE = '''"""
Auto-generated Dataset class for RelBench.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional

from plexe.relbench.base import Database, Dataset, Table


class GenDataset(Dataset):
    """Generated Dataset class for the relational database."""
    
    val_timestamp = pd.Timestamp("{val_timestamp}")
    test_timestamp = pd.Timestamp("{test_timestamp}")

    def __init__(self, csv_dir: str, cache_dir: Optional[str] = None):
        self.csv_dir = csv_dir
        super().__init__(cache_dir=cache_dir)

    def make_db(self) -> Database:
        path = self.csv_dir
        
{load_code}

{cleaning_code}

{time_processing_code}

        db = Database(
            table_dict={{
{table_dict_code}
            }}
        )
        
        return db
'''


class DatasetBuilderAgent(BaseAgent):
    """Agent for building RelBench Dataset classes from CSV data."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize the dataset builder agent.
        
        Args:
            config: Agent configuration
            additional_tools: Additional tools beyond defaults
        """
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
            context_parts.append(f"CSV files directory: {state['csv_dir']}")
        
        if state.get("schema_info"):
            schema = state["schema_info"]
            tables = list(schema.get("tables", {}).keys())
            context_parts.append(f"Tables from schema: {', '.join(tables)}")
            
            if schema.get("relationships"):
                rels = [f"{r['source_table']}.{r['source_column']} -> {r['target_table']}" 
                        for r in schema["relationships"]]
                context_parts.append(f"Relationships: {'; '.join(rels)}")
            
            if schema.get("temporal_columns"):
                temporal = [f"{t}: {cols}" for t, cols in schema["temporal_columns"].items()]
                context_parts.append(f"Temporal columns: {'; '.join(temporal)}")
        
        # Include EDA information from previous phase
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
        
        task_instruction = """
Your task:
1. Analyze the CSV files using get_csv_files_info
2. Get temporal statistics using get_temporal_statistics (use EDA suggestions if available)
3. Generate a complete GenDataset class with proper:
   - Table loading
   - Data cleaning (address quality issues found in EDA)
   - Temporal processing
   - Schema definition (PKs, FKs, time_cols)
4. Register the code using register_dataset_code

The output file should be saved as 'dataset.py' in the working directory.
"""
        context_parts.append(task_instruction)
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and extract dataset information from tool calls."""
        base_result = super()._process_result(result, state)
        
        dataset_info = {}
        generated_code = state.get("generated_code", {})
        
        if "tool_calls" in result and result["tool_calls"]:
            for tool_call in result["tool_calls"]:
                tool_name = tool_call.get("name", "")
                tool_result = tool_call.get("result", {})
                
                if tool_name == "register_dataset_code" and tool_result.get("status") == "registered":
                    dataset_info["class_name"] = tool_result.get("class_name", "GenDataset")
                    dataset_info["file_path"] = tool_result.get("file_path")
                    if "code" in tool_result:
                        generated_code["dataset"] = tool_result["code"]
        
        if not dataset_info:
            dataset_info["class_name"] = "GenDataset"
        
        if dataset_info:
            base_result["dataset_info"] = dataset_info
        
        if generated_code:
            base_result["generated_code"] = generated_code
        
        base_result["current_phase"] = PipelinePhase.TASK_BUILDING.value
        
        return base_result
