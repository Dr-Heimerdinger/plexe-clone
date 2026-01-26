"""
Relational GNN Specialist Agent.

This agent generates and executes GNN training scripts using
the plexe.relbench.modeling modules.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.common import save_artifact
from plexe.langgraph.tools.gnn_specialist import generate_training_script
from plexe.langgraph.prompts.gnn_specialist import GNN_SPECIALIST_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class RelationalGNNSpecialistAgent(BaseAgent):
    """
    Agent for GNN training script generation with Training-Free HPO.
    
    This agent uses MCP (Model Context Protocol) to access external
    knowledge sources for hyperparameter optimization without training.
    MCP tools are loaded automatically via MCPManager in BaseAgent.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        # Core GNN-specific tools (non-MCP)
        tools = [
            generate_training_script,
            save_artifact,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        # MCP tools for HPO will be loaded automatically by BaseAgent
        # via MCPManager.initialize() which includes:
        # - search_optimal_hyperparameters (from hpo-search MCP server)
        # - extract_hyperparameters_from_papers (from hpo-search MCP server)
        # - get_benchmark_hyperparameters (from hpo-search MCP server)
        # - compare_hyperparameter_configs (from hpo-search MCP server)
        
        super().__init__(
            agent_type="gnn_specialist",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return GNN_SPECIALIST_SYSTEM_PROMPT
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with training-specific information."""
        context_parts = []
        
        working_dir = state.get("working_dir", "")
        csv_dir = state.get("csv_dir", "")
        
        context_parts.append(f"Working directory: {working_dir}")
        context_parts.append(f"CSV directory: {csv_dir}")
        
        if state.get("dataset_info"):
            ds = state["dataset_info"]
            context_parts.append(f"Dataset file: {ds.get('file_path', working_dir + '/dataset.py')}")
            context_parts.append(f"Dataset class: {ds.get('class_name', 'GenDataset')}")
        
        if state.get("task_info"):
            task = state["task_info"]
            context_parts.append(f"Task file: {task.get('file_path', working_dir + '/task.py')}")
            context_parts.append(f"Task class: {task.get('class_name', 'GenTask')}")
            context_parts.append(f"Task type: {task.get('task_type', 'binary_classification')}")
        
        task_type = state.get("task_info", {}).get("task_type", "binary_classification")
        
        # Build dataset characteristics for HPO search
        dataset_chars = {
            "num_tables": len(state.get("schema_info", {}).get("tables", {})),
            "num_nodes": 10000,  # Estimate - would be calculated from schema
            "is_temporal": True,  # Always true for RelBench tasks
        }
        
        context_parts.append(f"""
EXECUTE THESE STEPS (Training-Free HPO via MCP):

1. SEARCH FOR OPTIMAL HYPERPARAMETERS using MCP tools:
   
   a) search_optimal_hyperparameters(
       task_type="{task_type}",
       num_nodes={dataset_chars.get('num_nodes', 10000)},
       num_tables={dataset_chars.get('num_tables', 5)},
       is_temporal={dataset_chars.get('is_temporal', True)},
       model_architecture="gnn"
   )
   # Returns: Heuristic-based hyperparameters with reasoning
   
   b) extract_hyperparameters_from_papers(
       paper_query="Relational GNN {task_type} temporal graphs",
       model_type="gnn",
       num_papers=5
   )
   # Returns: Hyperparameters extracted from recent papers
   
   c) get_benchmark_hyperparameters(
       task_type="{task_type}",
       dataset_domain="relational",
       model_architecture="gnn"
   )
   # Returns: Benchmark-proven hyperparameters
   
   d) compare_hyperparameter_configs(
       configs=[results_from_a, results_from_b, results_from_c],
       strategy="ensemble_median"
   )
   # Returns: Final recommended hyperparameters via ensemble voting

2. GENERATE TRAINING SCRIPT with optimal hyperparameters:
   generate_training_script(
       dataset_module_path="{working_dir}/dataset.py",
       dataset_class_name="GenDataset",
       task_module_path="{working_dir}/task.py",
       task_class_name="GenTask",
       working_dir="{working_dir}",
       task_type="{task_type}",
       **recommended_hyperparameters  # Use result from step 1d
   )

3. Report the selected hyperparameters with reasoning from all sources

NOTE: 
- All HPO tools are provided via MCP (Model Context Protocol)
- Training execution will be handled by the Operation Agent
- Focus on selecting optimal hyperparameters WITHOUT training experiments
""")
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and prepare for operation phase."""
        base_result = super()._process_result(result, state)
        
        import os
        
        working_dir = state.get("working_dir", "")
        script_path = os.path.join(working_dir, "train_script.py")
        
        # Check if training script was generated
        if os.path.exists(script_path):
            base_result["training_script_ready"] = True
            base_result["training_script_path"] = script_path
            logger.info(f"Training script generated at {script_path}")
        else:
            logger.warning("Training script not found")
            base_result["training_script_ready"] = False
        
        # Store selected hyperparameters in state for Operation Agent
        if "hyperparameters" in result:
            base_result["selected_hyperparameters"] = result["hyperparameters"]
        
        # Transition to OPERATION phase for execution
        base_result["current_phase"] = PipelinePhase.OPERATION.value
        
        return base_result
