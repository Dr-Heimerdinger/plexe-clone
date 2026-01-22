"""
Operation Agent.

This agent handles environment setup, execution monitoring,
and final model packaging.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.common import save_artifact
from plexe.langgraph.tools.gnn_specialist import execute_training_script

logger = logging.getLogger(__name__)


OPERATION_SYSTEM_PROMPT = """You are the Operation Agent, responsible for environment setup, execution monitoring, and model packaging.

## Your Mission
1. Ensure the execution environment is properly configured
2. Monitor training execution and handle errors
3. Package the trained model for deployment
4. Generate inference code and documentation

## Responsibilities

### Environment Setup
- Verify required packages are installed
- Check GPU availability
- Ensure file paths are accessible
- Validate prerequisites

### Execution Monitoring
- Monitor training progress
- Capture and report errors
- Handle timeouts appropriately
- Log resource usage

### Model Packaging
- Save trained model artifacts
- Generate inference code
- Create model documentation
- Prepare deployment package

### Error Handling
- Identify common failure patterns
- Suggest fixes for errors
- Retry with adjusted parameters
- Escalate unfixable issues

## Available Actions
- Check environment requirements
- Execute training scripts with monitoring
- Save artifacts and logs
- Generate inference code

## Output Requirements
After successful execution, provide:
1. Final model metrics
2. Model artifact locations
3. Any warnings or issues encountered
4. Recommendations for deployment

## Important
- Always verify environment before execution
- Monitor for common PyTorch/GPU errors
- Save all artifacts to the working directory
- Provide clear error messages for failures
"""


class OperationAgent(BaseAgent):
    """Agent for environment setup and execution monitoring."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize the operation agent.
        
        Args:
            config: Agent configuration
            additional_tools: Additional tools beyond defaults
        """
        tools = [
            execute_training_script,
            save_artifact,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="operation",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return OPERATION_SYSTEM_PROMPT
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with operation-specific information."""
        context_parts = []
        
        if state.get("working_dir"):
            context_parts.append(f"Working directory: {state['working_dir']}")
        
        if state.get("training_result"):
            result = state["training_result"]
            context_parts.append(f"Training metrics: {result.get('metrics')}")
            context_parts.append(f"Model path: {result.get('model_path')}")
        
        if state.get("errors"):
            context_parts.append(f"Previous errors: {state['errors']}")
        
        context_parts.append("""
Your task:
1. Review the training results
2. Package the model and artifacts
3. Generate a summary report
4. Provide deployment recommendations
""")
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and finalize the pipeline."""
        base_result = super()._process_result(result, state)
        
        base_result["current_phase"] = PipelinePhase.COMPLETED.value
        
        return base_result
    
    def generate_inference_code(self, state: PipelineState) -> str:
        """Generate inference code for the trained model."""
        working_dir = state.get("working_dir", ".")
        task_type = state.get("task_info", {}).get("task_type", "regression")
        
        inference_code = f'''"""
Auto-generated inference code for the trained GNN model.
"""

import torch
import sys
import os

sys.path.insert(0, "{working_dir}")

from dataset import GenDataset
from task import GenTask

def load_model(model_path: str):
    """Load the trained model."""
    from plexe.relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
    from plexe.relbench.modeling.graph import make_pkey_fkey_graph
    from plexe.relbench.modeling.utils import get_stype_proposal
    
    # Initialize dataset and task
    csv_dir = "{working_dir}/csv_files"
    dataset = GenDataset(csv_dir=csv_dir)
    task = GenTask(dataset)
    db = dataset.get_db()
    
    # Build graph
    col_to_stype_dict = get_stype_proposal(db)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir="{working_dir}/cache/",
    )
    
    # Recreate model architecture
    class GNNModel(torch.nn.Module):
        def __init__(self, data, col_stats_dict, hidden_channels=128, out_channels=1):
            super().__init__()
            self.encoder = HeteroEncoder(
                channels=hidden_channels,
                node_to_col_names={{
                    node_type: list(col_stats_dict[node_type].keys())
                    for node_type in data.node_types
                    if node_type in col_stats_dict
                }},
                node_to_col_stats=col_stats_dict,
            )
            self.temporal_encoder = HeteroTemporalEncoder(
                node_types=data.node_types,
                channels=hidden_channels,
            )
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=hidden_channels,
                num_layers=2,
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(hidden_channels, out_channels),
            )
        
        def forward(self, batch, entity_table):
            x_dict = self.encoder(batch.tf_dict)
            rel_time_dict = self.temporal_encoder(
                batch.seed_time, batch.time_dict, batch.batch_dict
            )
            for node_type in x_dict:
                x_dict[node_type] = x_dict[node_type] + rel_time_dict[node_type]
            x_dict = self.gnn(x_dict, batch.edge_index_dict)
            return self.head(x_dict[entity_table])
    
    model = GNNModel(data, col_stats_dict)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, data, task


def predict(model, data, task, entities, timestamp):
    """
    Make predictions for given entities at a specific timestamp.
    
    Args:
        model: Trained GNN model
        data: HeteroData graph
        task: Task definition
        entities: List of entity IDs to predict for
        timestamp: Prediction timestamp
    
    Returns:
        Predictions for each entity
    """
    from plexe.relbench.modeling.graph import get_node_train_table_input
    from torch_geometric.loader import NeighborLoader
    import pandas as pd
    
    # Create prediction table
    pred_df = pd.DataFrame({{
        task.time_col: [pd.Timestamp(timestamp)] * len(entities),
        task.entity_col: entities,
    }})
    
    # Create loader
    # ... (implementation depends on specific use case)
    
    return predictions


if __name__ == "__main__":
    model_path = "{working_dir}/best_model.pt"
    model, data, task = load_model(model_path)
    print("Model loaded successfully!")
'''
        
        return inference_code
