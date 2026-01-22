"""
Task Builder Agent.

This agent builds RelBench Task objects for prediction tasks,
generating SQL queries and complete Python Task classes.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.common import save_artifact
from plexe.langgraph.tools.dataset_builder import get_csv_files_info
from plexe.langgraph.tools.task_builder import test_sql_query, register_task_code

logger = logging.getLogger(__name__)


TASK_BUILDER_SYSTEM_PROMPT = """You are the Task Builder Agent, an expert in defining prediction tasks for relational databases.

## Your Mission
Create a Python Task class (GenTask) that defines:
1. The prediction target and entity
2. SQL queries to compute labels from historical data
3. Proper temporal windows for train/val/test splits
4. Evaluation metrics for the task

## Workflow

### Step 1: Understand the Prediction Task
Based on the user's intent, determine:
- Entity table (e.g., users, drivers, products)
- Target variable (e.g., churn, engagement, sales)
- Task type (e.g., regression, binary_classification, multiclass_classification)

### Step 2: Design SQL Query
Create a SQL query that:
- Uses temporal windows to prevent data leakage
- Computes labels from future data relative to timestamp
- Filters to relevant entities

### Step 3: Test the SQL Query
Use test_sql_query to validate your query works correctly.

### Step 4: Generate Task Code
Create a complete GenTask class:

```python
import pandas as pd
from plexe.relbench.base import EntityTask, RecommendationTask
from plexe.relbench.metrics import accuracy, mae, rmse, f1, auroc

class GenTask(EntityTask):  # or RecommendationTask
    entity_table = "entity_table_name"
    entity_col = "entity_id_column"
    time_col = "timestamp"
    target_col = "target"
    task_type = "regression"  # or binary_classification, multiclass_classification
    timedelta = pd.Timedelta(days=30)
    metrics = [mae, rmse]  # appropriate metrics

    def make_table(self, db, timestamps):
        # SQL query using DuckDB
        query = '''
        SELECT
            t.timestamp as {time_col},
            e.{entity_col},
            -- aggregation to compute target
        FROM timestamp_df t
        LEFT JOIN {entity_table} e ON ...
        LEFT JOIN {activity_table} a ON ...
        WHERE temporal_filter
        GROUP BY t.timestamp, e.{entity_col}
        '''
        
        import duckdb
        conn = duckdb.connect()
        # Register tables from db
        for table_name, table in db.table_dict.items():
            conn.register(table_name, table.df)
        conn.register("timestamp_df", timestamps)
        
        df = conn.execute(query).fetchdf()
        return df
```

### Step 5: Register the Code
Use register_task_code to save the generated code.

## SQL Query Patterns

### Binary Classification (Churn-style)
```sql
SELECT t.timestamp as timestamp, e.user_id,
       CASE WHEN NOT EXISTS (
           SELECT 1 FROM activity a
           WHERE a.user_id = e.user_id
           AND a.time > t.timestamp
           AND a.time <= t.timestamp + INTERVAL '{timedelta}'
       ) THEN 1 ELSE 0 END as target
FROM timestamp_df t, users e
WHERE e.created_at <= t.timestamp
```

### Regression (Count-style)
```sql
SELECT t.timestamp, e.entity_id,
       COUNT(a.id) as target
FROM timestamp_df t
CROSS JOIN entities e
LEFT JOIN activities a ON a.entity_id = e.id
    AND a.time > t.timestamp
    AND a.time <= t.timestamp + INTERVAL '{timedelta}'
WHERE e.created_at <= t.timestamp
GROUP BY t.timestamp, e.entity_id
```

### Classification (Threshold-style)
```sql
SELECT t.timestamp, e.entity_id,
       CASE WHEN MIN(a.position) <= 3 THEN 1 ELSE 0 END as target
FROM timestamp_df t
LEFT JOIN results a ON a.entity_id = e.id
    AND a.time > t.timestamp
    AND a.time <= t.timestamp + INTERVAL '{timedelta}'
WHERE e.id IN (SELECT DISTINCT entity_id FROM results WHERE time > t.timestamp - INTERVAL '1 year')
GROUP BY t.timestamp, e.entity_id
```

## Available Metrics
- Regression: mae, rmse, r2
- Binary Classification: accuracy, f1, auroc, average_precision
- Multiclass Classification: accuracy, f1_macro, f1_micro

## Important
- Use class name 'GenTask'
- Always test SQL before finalizing
- Use exact column names from the database (snake_case)
- Ensure temporal windows prevent data leakage
- Save as 'task.py' in the working directory
"""


class TaskBuilderAgent(BaseAgent):
    """Agent for building RelBench Task classes."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize the task builder agent.
        
        Args:
            config: Agent configuration
            additional_tools: Additional tools beyond defaults
        """
        tools = [
            get_csv_files_info,
            test_sql_query,
            register_task_code,
            save_artifact,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="task_builder",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return TASK_BUILDER_SYSTEM_PROMPT
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with task-specific information including EDA results."""
        context_parts = []
        
        if state.get("working_dir"):
            context_parts.append(f"Working directory: {state['working_dir']}")
        
        if state.get("csv_dir"):
            context_parts.append(f"CSV files directory: {state['csv_dir']}")
        
        if state.get("user_intent"):
            context_parts.append(f"User intent: {state['user_intent']}")
        
        if state.get("schema_info"):
            schema = state["schema_info"]
            tables = list(schema.get("tables", {}).keys())
            context_parts.append(f"Available tables: {', '.join(tables)}")
            
            for table_name, table_info in schema.get("tables", {}).items():
                pk = table_info.get("primary_key", [])
                if pk:
                    context_parts.append(f"  {table_name} PK: {pk}")
        
        if state.get("dataset_info"):
            ds = state["dataset_info"]
            context_parts.append(f"Dataset class: {ds.get('class_name')}")
            if ds.get("val_timestamp"):
                context_parts.append(f"Val timestamp: {ds.get('val_timestamp')}")
            if ds.get("test_timestamp"):
                context_parts.append(f"Test timestamp: {ds.get('test_timestamp')}")
        
        # Include EDA information for better task design
        if state.get("eda_info"):
            eda = state["eda_info"]
            context_parts.append("\n## EDA Analysis Results:")
            
            if eda.get("statistics"):
                context_parts.append("Table Statistics:")
                for table, stats in eda["statistics"].items():
                    if isinstance(stats, dict):
                        row_count = stats.get("row_count", "unknown")
                        context_parts.append(f"  - {table}: {row_count} rows")
            
            if eda.get("temporal_analysis"):
                context_parts.append("Temporal Analysis:")
                for table, analysis in eda["temporal_analysis"].items():
                    if analysis.get("time_columns"):
                        context_parts.append(f"  - {table}: time columns = {analysis['time_columns']}")
                    if analysis.get("date_range"):
                        context_parts.append(f"    Date range: {analysis['date_range']}")
            
            if eda.get("relationship_analysis"):
                context_parts.append("Relationship Analysis:")
                for key, info in eda["relationship_analysis"].items():
                    if isinstance(info, dict):
                        if info.get("cardinality"):
                            context_parts.append(f"  - {key}: {info['cardinality']}")
                        if info.get("join_quality"):
                            context_parts.append(f"    Join quality: {info['join_quality']}")
            
            if eda.get("summary"):
                context_parts.append(f"\nEDA Summary: {eda['summary']}")
        
        task_instruction = """
Your task:
1. Analyze the schema and user intent
2. Use EDA insights to understand data patterns and relationships
3. Design a SQL query for label computation
4. Test the SQL query using test_sql_query
5. Generate a complete GenTask class
6. Register the code using register_task_code

The output file should be saved as 'task.py' in the working directory.
"""
        context_parts.append(task_instruction)
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and extract task information from tool calls."""
        base_result = super()._process_result(result, state)
        
        task_info = {}
        generated_code = state.get("generated_code", {})
        
        if "tool_calls" in result and result["tool_calls"]:
            for tool_call in result["tool_calls"]:
                tool_name = tool_call.get("name", "")
                tool_result = tool_call.get("result", {})
                
                if tool_name == "register_task_code" and tool_result.get("status") == "registered":
                    task_info["class_name"] = tool_result.get("class_name", "GenTask")
                    task_info["file_path"] = tool_result.get("file_path")
                    task_info["task_type"] = tool_result.get("task_type")
                    if "code" in tool_result:
                        generated_code["task"] = tool_result["code"]
        
        if not task_info:
            task_info["class_name"] = "GenTask"
        
        if task_info:
            base_result["task_info"] = task_info
        
        if generated_code:
            base_result["generated_code"] = generated_code
        
        base_result["current_phase"] = PipelinePhase.GNN_TRAINING.value
        
        return base_result
