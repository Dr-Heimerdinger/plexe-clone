# Plexe LangGraph Multi-Agent System

This document describes the LangGraph-based multi-agent system for building ML models from relational databases.

## Architecture Overview

The system consists of specialized agents coordinated by a LangGraph StateGraph workflow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PlexeOrchestrator                            │
│                     (LangGraph StateGraph)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │Conversational│───▶│   Graph      │───▶│   Dataset    │         │
│   │    Agent     │    │  Architect   │    │   Builder    │         │
│   └──────────────┘    └──────────────┘    └──────────────┘         │
│          │                   │                   │                  │
│          │                   │                   ▼                  │
│          │                   │            ┌──────────────┐         │
│          │                   │            │    Task      │         │
│          │                   │            │   Builder    │         │
│          │                   │            └──────────────┘         │
│          │                   │                   │                  │
│          │                   │                   ▼                  │
│          │                   │            ┌──────────────┐         │
│          │                   │            │     GNN      │         │
│          │                   │            │  Specialist  │         │
│          │                   │            └──────────────┘         │
│          │                   │                   │                  │
│          │                   │                   ▼                  │
│          │                   │            ┌──────────────┐         │
│          └───────────────────┴───────────▶│  Operation   │         │
│                                           │    Agent     │         │
│                                           └──────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Agent Descriptions

### 1. Conversational Agent
**Role**: User interaction and requirements gathering

**Responsibilities**:
- Guide users through ML problem definition
- Validate data availability (CSV or database)
- Extract prediction targets and entity types
- Get explicit user confirmation before proceeding

**Tools**:
- `get_dataset_preview`: Preview CSV data
- `validate_db_connection`: Test database connectivity

### 2. Relational Graph Architect Agent
**Role**: Schema analysis and data export

**Responsibilities**:
- Connect to databases and analyze schema
- Identify primary keys, foreign keys, and temporal columns
- Classify tables (Fact vs Dimension)
- Export all tables to CSV format

**Tools**:
- `validate_db_connection`: Verify database access
- `export_tables_to_csv`: Export data to CSV
- `extract_schema_metadata`: Analyze schema structure

### 3. Dataset Builder Agent
**Role**: Generate RelBench Dataset classes

**Responsibilities**:
- Analyze CSV files and schema metadata
- Determine temporal split timestamps
- Generate complete Python Dataset class (GenDataset)
- Handle data cleaning and preprocessing

**Tools**:
- `get_csv_files_info`: List CSV files
- `get_temporal_statistics`: Analyze timestamps
- `register_dataset_code`: Save generated code

**Output**: `dataset.py` with GenDataset class

### 4. Task Builder Agent
**Role**: Generate RelBench Task classes

**Responsibilities**:
- Understand prediction requirements
- Design SQL queries for label computation
- Generate complete Python Task class (GenTask)
- Define metrics and evaluation

**Tools**:
- `get_csv_files_info`: Understand data structure
- `test_sql_query`: Validate SQL queries
- `register_task_code`: Save generated code

**Output**: `task.py` with GenTask class

### 5. Relational GNN Specialist Agent
**Role**: GNN training script generation and execution

**Responsibilities**:
- Generate training scripts using plexe.relbench.modeling
- Configure HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
- Execute training with temporal sampling
- Report model performance metrics

**Tools**:
- `generate_training_script`: Create train_script.py
- `execute_training_script`: Run training

**Output**: `train_script.py`, `best_model.pt`, `training_results.json`

### 6. Operation Agent
**Role**: Environment management and finalization

**Responsibilities**:
- Verify environment requirements
- Monitor training execution
- Handle errors and retries
- Generate inference code
- Package model artifacts

**Tools**:
- `execute_training_script`: Run scripts
- `save_artifact`: Save files

## Pipeline Flow

### Phase 1: Conversation
```
User Message → Conversational Agent → Requirements Gathered
                     │
                     ▼
            User Confirmation Required?
                     │
              Yes ───┴─── No
               │          │
               ▼          └──▶ Continue Conversation
         Proceed to Pipeline
```

### Phase 2: Schema Analysis
```
Database Connection → Graph Architect Agent → Schema Metadata
                           │
                           ▼
                    Export Tables to CSV
                           │
                           ▼
                    Register in ObjectRegistry
```

### Phase 3: Dataset Building
```
CSV Files + Schema → Dataset Builder Agent → GenDataset Class
                           │
                           ▼
                    Analyze Temporal Data
                           │
                           ▼
                    Generate dataset.py
```

### Phase 4: Task Building
```
Schema + User Intent → Task Builder Agent → GenTask Class
                             │
                             ▼
                      Design SQL Query
                             │
                             ▼
                      Test and Validate
                             │
                             ▼
                      Generate task.py
```

### Phase 5: GNN Training
```
dataset.py + task.py → GNN Specialist Agent → Trained Model
                             │
                             ▼
                      Generate train_script.py
                             │
                             ▼
                      Execute Training
                             │
                             ▼
                      Report Metrics
```

### Phase 6: Operation
```
Training Results → Operation Agent → Final Package
                        │
                        ▼
                 Generate Inference Code
                        │
                        ▼
                 Package Artifacts
```

## Configuration

### Environment Variables

Configure each agent's model via `.env`:

```bash
# API Keys
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
GEMINI_API_KEY=your-key

# Agent Models (format: provider/model-name)
PLEXE_ORCHESTRATOR_MODEL="openai/gpt-4o"
PLEXE_CONVERSATIONAL_MODEL="openai/gpt-4o"
PLEXE_GRAPH_ARCHITECT_MODEL="openai/gpt-4o"
PLEXE_DATASET_BUILDER_MODEL="openai/gpt-4o"
PLEXE_TASK_BUILDER_MODEL="openai/gpt-4o"
PLEXE_GNN_SPECIALIST_MODEL="openai/gpt-4o"
PLEXE_OPERATION_MODEL="openai/gpt-4o"

# Agent Settings
PLEXE_AGENT_TEMPERATURE="0.1"
PLEXE_MAX_RETRIES="3"
PLEXE_VERBOSE="false"
```

### Supported Model Formats
- OpenAI: `openai/gpt-4o`, `openai/gpt-4-turbo`
- Anthropic: `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-3-opus`
- Google: `gemini/gemini-2.5-flash`, `gemini/gemini-pro`

## Usage

### Basic Usage

```python
from plexe import PlexeOrchestrator

orchestrator = PlexeOrchestrator(verbose=True)

result = orchestrator.run(
    user_message="Build a model to predict user churn",
    db_connection_string="postgresql://user:pass@localhost:5432/mydb",
)

print(f"Status: {result['status']}")
print(f"Working directory: {result['working_dir']}")
```

### Interactive Chat

```python
from plexe import PlexeOrchestrator

orchestrator = PlexeOrchestrator()

# Start session
result = orchestrator.run(
    user_message="I want to predict customer churn",
    session_id="my-session"
)

# Continue conversation
response = orchestrator.chat(
    message="The data is in a PostgreSQL database",
    session_id="my-session"
)
```

### Custom Configuration

```python
from plexe import PlexeOrchestrator, AgentConfig

config = AgentConfig(
    orchestrator_model="anthropic/claude-sonnet-4-20250514",
    conversational_model="openai/gpt-4o",
    gnn_specialist_model="openai/gpt-4o",
    temperature=0.2,
    max_retries=5,
)

orchestrator = PlexeOrchestrator(config=config)
```

## State Management

The pipeline uses a shared `PipelineState` that flows between agents:

```python
class PipelineState(TypedDict):
    session_id: str
    working_dir: str
    current_phase: str
    messages: List[Message]
    user_intent: str
    db_connection_string: Optional[str]
    csv_dir: Optional[str]
    schema_info: Optional[SchemaInfo]
    dataset_info: Optional[DatasetInfo]
    task_info: Optional[TaskInfo]
    training_result: Optional[TrainingResult]
    generated_code: Dict[str, str]
    artifacts: List[str]
    errors: List[str]
    warnings: List[str]
```

## Error Handling

The system includes automatic error recovery:

1. **Retry Mechanism**: Failed steps are retried up to `PLEXE_MAX_RETRIES` times
2. **Error Routing**: Errors route to the error handler node
3. **Graceful Degradation**: Pipeline can continue from last successful state

## Generated Artifacts

After successful execution, the working directory contains:

```
workdir/session-YYYYMMDD-HHMMSS/
├── csv_files/           # Exported table data
│   ├── users.csv
│   ├── orders.csv
│   └── ...
├── cache/               # Graph cache
├── dataset.py           # GenDataset class
├── task.py              # GenTask class
├── train_script.py      # GNN training script
├── best_model.pt        # Trained model weights
├── training_results.json # Metrics and metadata
└── inference_code.py    # Inference utilities
```

## Key Components

### plexe.relbench.modeling Modules

The GNN training uses these core modules:

- **make_pkey_fkey_graph**: Converts Database to HeteroData graph
- **HeteroEncoder**: Encodes tabular features to embeddings
- **HeteroTemporalEncoder**: Encodes temporal information
- **HeteroGraphSAGE**: GNN message passing layers
- **NeighborLoader**: Temporal-aware batch sampling

### ObjectRegistry

Shared state storage for cross-agent communication:

```python
from plexe.core.object_registry import ObjectRegistry

registry = ObjectRegistry()
registry.register(str, "schema_metadata", data)
data = registry.get(dict, "schema_metadata")
```

## Extension Points

### Adding Custom Tools

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(param: str) -> dict:
    """Description of what the tool does."""
    return {"result": "value"}

# Add to agent
agent = DatasetBuilderAgent(additional_tools=[my_custom_tool])
```

### Custom Agents

```python
from plexe.langgraph.agents.base import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, config=None):
        super().__init__(
            agent_type="custom",
            config=config,
            tools=[...],
        )
    
    @property
    def system_prompt(self) -> str:
        return "Your custom system prompt..."
```

## Performance Considerations

- **Model Selection**: Use faster models (gpt-4o-mini) for simpler agents
- **Batch Size**: Adjust based on GPU memory (default: 512)
- **Epochs**: Start with 10 epochs, increase if needed
- **Caching**: Graph construction is cached for repeated runs

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify connection string format
   - Check network access to database
   - Ensure credentials are correct

2. **Training OOM Error**
   - Reduce batch_size
   - Reduce num_neighbors
   - Use CPU if GPU memory insufficient

3. **SQL Query Errors**
   - Verify column names (snake_case in PostgreSQL)
   - Test queries with test_sql_query tool
   - Check temporal window logic

4. **Missing Dependencies**
   - Run `poetry install` to install all dependencies
   - Ensure LangGraph packages are installed
