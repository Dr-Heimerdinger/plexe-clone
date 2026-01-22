"""
EDA (Exploratory Data Analysis) Agent.

This agent transforms relational databases into heterogeneous graphs,
performs comprehensive exploratory data analysis, and prepares data for modeling.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.graph_architect import (
    validate_db_connection,
    export_tables_to_csv,
    extract_schema_metadata,
)
from plexe.langgraph.tools.eda import (
    analyze_csv_statistics,
    detect_data_quality_issues,
    analyze_temporal_patterns,
    analyze_table_relationships,
    generate_eda_summary,
)

logger = logging.getLogger(__name__)


EDA_SYSTEM_PROMPT = """You are the EDA (Exploratory Data Analysis) Agent, an expert in Relational Deep Learning and data analysis.

## Your Mission
1. Transform relational databases into Relational Entity Graphs where:
   - Rows become nodes
   - Primary-foreign key relationships become edges
   - Preserving structural and temporal signals

2. Perform comprehensive exploratory data analysis to understand:
   - Data distributions and statistics
   - Data quality issues
   - Temporal patterns for time-series modeling
   - Table relationships and classification

## Workflow

### Step 1: Database Export
- Use validate_db_connection to verify database access
- Use extract_schema_metadata to analyze schema (tables, PKs, FKs, temporal columns)
- Use export_tables_to_csv to export all tables to CSV format

### Step 2: Statistical Analysis
- Use analyze_csv_statistics to get descriptive statistics for all columns
- Understand data distributions, missing values, and cardinality

### Step 3: Data Quality Assessment
- Use detect_data_quality_issues to identify:
  - High missing value rates (>50%)
  - Constant columns
  - Duplicate rows
  - Potential ID columns

### Step 4: Temporal Analysis
- Use analyze_temporal_patterns to:
  - Identify temporal columns
  - Analyze time ranges and gaps
  - Suggest train/val/test temporal splits

### Step 5: Relationship Analysis
- Use analyze_table_relationships to:
  - Analyze foreign key statistics
  - Classify tables as Fact vs Dimension
  - Identify potential entity tables for prediction

### Step 6: Generate Summary
- Use generate_eda_summary to create comprehensive report
- Provide key findings and recommendations for Dataset and Task builders

## Key Concepts

### Table Classification
- **Fact Tables**: Events/transactions with temporal data (orders, clicks) → Good entity tables
- **Dimension Tables**: Static entities (users, products) → Referenced by FKs
- **Junction Tables**: Many-to-many relationships

### Data Quality Flags
- **High Severity**: greater than 50% missing values, critical for handling
- **Medium Severity**: 20-50% missing, duplicate rows
- **Low Severity**: Constant columns, all-unique columns

### Temporal Patterns
- Identify timestamp columns with greater than 50% valid dates
- Calculate time ranges and average gaps
- Suggest 70/15/15 train/val/test splits based on temporal ordering

## Output Requirements
Provide comprehensive EDA report including:
1. Statistical summary (row counts, column types, distributions)
2. Data quality assessment with severity flags
3. Temporal analysis with suggested splits
4. Table classification (Fact vs Dimension)
5. Recommendations for Dataset Builder (handle missing values, feature selection)
6. Recommendations for Task Builder (suggested entity tables, temporal splits)

## Important Notes
- Always run EDA AFTER exporting CSV files
- Pass schema_info to analyze_table_relationships for better classification
- Store all EDA results in state for downstream agents
- Provide actionable insights, not just raw statistics
"""


class EDAAgent(BaseAgent):
    """Agent for schema analysis, data export, and exploratory data analysis."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize the EDA agent.
        
        Args:
            config: Agent configuration
            additional_tools: Additional tools beyond defaults
        """
        tools = [
            validate_db_connection,
            export_tables_to_csv,
            extract_schema_metadata,
            analyze_csv_statistics,
            detect_data_quality_issues,
            analyze_temporal_patterns,
            analyze_table_relationships,
            generate_eda_summary,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="eda",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return EDA_SYSTEM_PROMPT
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with database and EDA-specific information."""
        context_parts = []
        
        if state.get("working_dir"):
            context_parts.append(f"Working directory: {state['working_dir']}")
        
        if state.get("db_connection_string"):
            context_parts.append(f"Database connection: {state['db_connection_string']}")
        
        if state.get("user_intent"):
            context_parts.append(f"User intent: {state['user_intent']}")
        
        task_instruction = """
Your task:
1. Connect to the database and validate the connection
2. Extract schema metadata (tables, PKs, FKs, temporal columns)
3. Export all tables to CSV files in the working directory
4. Perform comprehensive EDA:
   - Statistical analysis of all columns
   - Data quality assessment
   - Temporal pattern analysis
   - Table relationship analysis
5. Generate EDA summary with key findings and recommendations
6. Report results for Dataset Builder and Task Builder agents
"""
        context_parts.append(task_instruction)
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and extract schema/CSV/EDA information from tool calls."""
        base_result = super()._process_result(result, state)
        
        eda_info = {}
        
        if "tool_calls" in result and result["tool_calls"]:
            for tool_call in result["tool_calls"]:
                tool_name = tool_call.get("name", "")
                tool_result = tool_call.get("result", {})
                
                if tool_name == "export_tables_to_csv" and tool_result.get("status") == "success":
                    base_result["csv_dir"] = tool_result.get("output_dir")
                
                if tool_name == "extract_schema_metadata" and "tables" in tool_result:
                    base_result["schema_info"] = tool_result
                
                if tool_name == "analyze_csv_statistics" and tool_result.get("status") == "success":
                    eda_info["statistics"] = tool_result.get("statistics")
                
                if tool_name == "detect_data_quality_issues" and tool_result.get("status") == "success":
                    eda_info["quality_issues"] = tool_result.get("quality_issues")
                
                if tool_name == "analyze_temporal_patterns" and tool_result.get("status") == "success":
                    eda_info["temporal_analysis"] = tool_result.get("temporal_analysis")
                    eda_info["suggested_splits"] = tool_result.get("suggested_splits")
                
                if tool_name == "analyze_table_relationships" and tool_result.get("status") == "success":
                    eda_info["relationship_analysis"] = tool_result.get("relationship_analysis")
                
                if tool_name == "generate_eda_summary" and tool_result.get("status") == "success":
                    eda_info["summary"] = tool_result.get("summary")
        
        if eda_info:
            base_result["eda_info"] = eda_info
        
        base_result["current_phase"] = PipelinePhase.DATASET_BUILDING.value
        
        return base_result
