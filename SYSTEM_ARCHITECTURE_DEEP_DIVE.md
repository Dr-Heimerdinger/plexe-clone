# Hệ Thống Đa Tác Tử Plexe - Phân Tích Toàn Diện

## 1. TỔNG QUAN HỆ THỐNG

Plexe là một hệ thống **tự động xây dựng mô hình ML** sử dụng **arsitektur đa tác tử**. Hệ thống sử dụng LLM để tự động sinh code, training, evaluation, và deployment.

### Kiến Trúc Cấp Cao
```
User Intent + Datasets/DB → ModelBuilder → PlexeAgent (Orchestrator) → Trained Model + Inference Code
```

---

## 2. LUỒNG XỬ LÝ CHÍNH

### 2.1 Bước 1: Khởi Tạo (ModelBuilder.build)
**File**: `plexe/model_builder.py`

```
ModelBuilder.build() {
  1. Kiểm tra input (datasets hoặc db_connection_string)
  2. Tạo ObjectRegistry (registry toàn cục)
  3. Đăng ký datasets hoặc connection string
  4. Tạo PlexeAgent (orchestrator)
  5. Chạy PlexeAgent.run()
}
```

**Output**: ModelGenerationResult (training code, inference code, predictor)

---

## 3. PLEXE AGENT - ORCHESTRATOR

**File**: `plexe/agents/agents.py`

PlexeAgent là **orchestrator** chính điều phối tất cả các agents khác. Nó tạo một hệ thống các agents có công khai:

### 3.1 Các Agents Chính

| Agent | File | Mục Đích | Code? |
|-------|------|---------|-------|
| **MLResearchScientist** | `model_planner.py` | Lập kế hoạch ML solutions | ✓ ToolCallingAgent |
| **FeatureEngineer** | `feature_engineer.py` | Transform raw data thành features | ✓ CodeAgent |
| **DatasetSplitter** | `dataset_splitter.py` | Split datasets thành train/val | ✓ CodeAgent |
| **MLEngineer** | `model_trainer.py` | Implement training code | ✓✓ **Main Code Gen** |
| **MLOperationsEngineer** | `model_packager.py` | Generate inference code | ✓✓ **Main Code Gen** |
| **ModelTester** | `model_tester.py` | Test & evaluate model | ✓ CodeAgent |
| **RelationalGraphArchitect** | `relational_graph_architect.py` | Transform DB → Graph | ✓ CodeAgent (RDL) |
| **TemporalTaskSupervisor** | `temporal_task_supervisor.py` | Handle temporal tasks | ✓ CodeAgent (RDL) |
| **RelationalGNNSpecialist** | `relational_gnn_specialist.py` | Train GNNs on graphs | ✓ CodeAgent (RDL) |

---

## 4. CHI TIẾT: AGENTS SINH CODE

### 4.1 MLEngineer (Model Trainer Agent) - **SINH TRAINING CODE**

**File**: `plexe/agents/model_trainer.py`

Đây là agent **chính** sinh training code!

#### Công Cụ Sử Dụng:
```python
tools=[
    get_training_code_generation_tool(tool_model_id),  # Sinh code
    validate_training_code,                             # Kiểm tra code
    get_executor_tool(distributed),                     # Thực thi code
    get_training_code_fixing_tool(tool_model_id),      # Fix lỗi
    get_dataset_schema,                                # Schema info
    get_training_datasets,                             # Lấy data
    ...
]
```

#### Quy Trình:
1. **Nhận input**:
   - ML task definition (intent)
   - Metric to optimize (name + comparison method)
   - Solution ID từ ML Research Scientist
   - Train/validation dataset names
   - Working directory

2. **Sinh training code**:
   ```python
   training_code = generate_training_code(
       task=intent,
       solution_plan=solution_plan,
       train_datasets=["dataset_0"],
       validation_datasets=["dataset_1"]
   )
   ```

3. **Validate code**:
   ```python
   validate_training_code(code)
   ```

4. **Execute code**:
   ```python
   execute_training_code(
       solution_id=solution_id,
       code=training_code,
       working_dir=working_dir,
       dataset_names=["dataset_0", "dataset_1"],
       timeout=1800,
       metric_to_optimise_name="accuracy",
       metric_to_optimise_comparison_method="HIGHER_IS_BETTER"
   )
   ```

5. **Lưu kết quả**:
   - Training code lưu vào Solution object
   - Model artifacts lưu vào ObjectRegistry
   - Performance metric tính toán

#### Code Generation Tool:
**File**: `plexe/tools/training.py`

```python
@tool
def generate_training_code(
    task: str, 
    solution_plan: str, 
    train_datasets: List[str], 
    validation_datasets: List[str]
) -> str:
    train_generator = TrainingCodeGenerator(Provider(llm_to_use))
    return train_generator.generate_training_code(...)
```

**Điểm chính**: Agent sử dụng **TrainingCodeGenerator** từ `plexe/internal/models/generation/training.py` để sinh code.

---

### 4.2 MLOperationsEngineer (Model Packager Agent) - **SINH INFERENCE CODE**

**File**: `plexe/agents/model_packager.py`

Sinh **production-ready inference code** từ training code.

#### Công Cụ:
```python
tools=[
    get_inference_context_tool(tool_model_id),  # Lấy context từ training
    validate_inference_code,
    list_solutions,
]
```

#### Quy Trình:
1. **Lấy training code** từ Solution
2. **Sinh inference wrapper**:
   - Lớp `PredictorImplementation`
   - Load model artifacts
   - Implement `predict()` method
3. **Validate inference code**
4. **Lưu vào Solution object**

---

### 4.3 FeatureEngineer - **SINH FEATURE TRANSFORMATION CODE**

**File**: `plexe/agents/feature_engineer.py`

Transform raw datasets.

#### Công Cụ:
```python
tools=[
    get_dataset_preview,
    validate_feature_transformations,
    apply_feature_transformer,
    get_dataset_reports,
]
```

#### Quy Trình:
1. Nhận EDA report
2. Sinh feature transformation code (pandas, sklearn)
3. Execute & lưu transformed datasets

---

### 4.4 Agents Relational Deep Learning (RDL)

Để xử lý **relational database** tasks:

#### RelationalGraphArchitect
**File**: `plexe/agents/relational_graph_architect.py`

**Mục đích**: Transform DB → Heterogeneous Graph

**Công Cụ**:
```python
tools=[
    extract_schema_metadata,           # Lấy schema từ DB
    get_table_columns,                # ⚠️ Kiểm tra column names!
    load_table_data,
    build_hetero_graph,               # Xây dựng graph
    encode_multi_modal_features,
    create_entity_mapper,             # Mapping entities
]
```

**Sinh code**: YES - Python code để xây dựng graph

---

#### TemporalTaskSupervisor
**File**: `plexe/agents/temporal_task_supervisor.py`

**Mục đích**: Định nghĩa temporal tasks, đảm bảo temporal consistency

**Công Cụ**:
```python
tools=[
    discover_temporal_columns,        # Tìm temporal columns
    execute_sql_query,               # Chạy SQL
    define_training_task,
    generate_sql_implementation,      # ⚠️ Sinh SQL!
    temporal_split,
    generate_temporal_splits_from_db,
]
```

**Sinh code**: YES - SQL code + Python code

---

#### RelationalGNNSpecialist
**File**: `plexe/agents/relational_gnn_specialist.py`

**Mục đích**: Train GNN trên graph

**Công Cụ**:
```python
tools=[
    get_hetero_graph_from_registry,
    load_training_data,
    configure_temporal_sampler,
    build_gnn_model,                 # Xây dựng model
    train_gnn_epoch,                 # Training loop
    evaluate_gnn,
    save_gnn_model,
]
```

**Sinh code**: YES - PyTorch code cho GNN

---

## 5. CHI TIẾT: CODE EXECUTION

### 5.1 Execution Tool

**File**: `plexe/tools/execution.py`

```python
@tool
def execute_training_code(
    solution_id: str,
    code: str,
    working_dir: str,
    dataset_names: List[str],
    timeout: int,
    metric_to_optimise_name: str,
    metric_to_optimise_comparison_method: str,
) -> Dict:
    """Thực thi training code trong isolated environment"""
    
    # 1. Lấy datasets từ ObjectRegistry
    datasets = {}
    for name in dataset_names:
        datasets[name] = object_registry.get(TabularConvertible, name)
    
    # 2. Tạo execution context
    exec_context = {
        'pd': pd,
        'np': np,
        'datasets': datasets,
        ...
    }
    
    # 3. Execute code
    exec(code, exec_context)
    
    # 4. Lấy kết quả (trainer, features, metrics)
    trainer = exec_context.get('trainer')
    features = exec_context.get('training_features')
    
    # 5. Lưu artifacts vào ObjectRegistry
    for artifact in artifacts:
        object_registry.register(Artifact, f"{solution_id}_{name}", artifact)
    
    # 6. Tính performance
    performance = evaluate_solution(...)
    
    return {
        'status': 'success',
        'performance': performance.value,
        'artifacts': artifact_ids,
    }
```

**Điểm chính**:
- Code chạy trong **isolated environment**
- Datasets lấy từ **ObjectRegistry**
- Artifacts lưu vào **ObjectRegistry**
- Có **timeout protection**

---

## 6. CHI TIẾT: TRAINING PROCESS

### 6.1 Quy Trình Training

```
1. MLResearchScientist {
   - Phân tích intent, input/output schema
   - Lập kế hoạch ML solutions
   → Solutions: ID + plan
}

2. FeatureEngineer {
   - Analyze EDA reports
   - Sinh feature transformation code
   → Transformed datasets
}

3. DatasetSplitter {
   - Split train/validation datasets
   → Split datasets
}

4. MLEngineer {
   ⭐ MAIN CODE GENERATION ⭐
   - Sinh training code
   - Execute code (training loop)
   - Compute performance
   → Training code + Model artifacts + Performance
}

5. MLOperationsEngineer {
   - Sinh inference code (wrapper)
   → Inference code
}

6. ModelTester {
   - Execute on test set
   - Generate evaluation report
   → Test performance + Report
}

7. Orchestrator {
   - Select best solution
   - Format final output
}
```

---

## 7. CHI TIẾT: EVALUATION

### 7.1 Performance Metrics

**File**: `plexe/tools/metrics.py`

```python
@tool
def get_select_target_metric(llm_to_use: str) -> str:
    """Lựa chọn metric phù hợp"""
    # Ví dụ: classification → accuracy
    # Ví dụ: regression → RMSE
```

### 7.2 Model Review

**File**: `plexe/tools/evaluation.py`

```python
@tool
def review_finalised_model(
    intent: str,
    solution_id: str,
) -> dict:
    """
    Review toàn bộ model:
    - Input/output schemas
    - Training code quality
    - Inference code quality
    - Expected performance
    """
    reviewer = ModelReviewer(Provider(llm_to_use))
    return reviewer.review_model(
        intent, 
        input_schema, 
        output_schema, 
        solution.plan, 
        solution.training_code, 
        solution.inference_code
    )
```

### 7.3 Solution Evaluation

```python
@tool
def get_solution_performances() -> Dict[str, float]:
    """Lấy performance của tất cả solutions"""
    # Returns: {
    #     "solution_id_1": 0.95,
    #     "solution_id_2": 0.92,
    # }
```

---

## 8. OBJECT REGISTRY - SHARED STATE

**File**: `plexe/core/object_registry.py`

Hệ thống sử dụng **ObjectRegistry** (singleton) để chia sẻ data giữa agents:

```python
object_registry = ObjectRegistry()

# Register
object_registry.register(TabularConvertible, "dataset_0", df)
object_registry.register(Solution, "solution_1", solution)
object_registry.register(Artifact, "artifact_model", model.pkl)

# Retrieve
dataset = object_registry.get(TabularConvertible, "dataset_0")
solution = object_registry.get(Solution, "best_performing_solution")

# List
all_solutions = object_registry.list_by_type(Solution)
```

**Tất cả agents** đều có access vào registry này!

---

## 9. RELATIONAL DEEP LEARNING (RDL) PIPELINE

Khi `db_connection_string` được cung cấp:

### 9.1 Temporal Task Definition
```
TemporalTaskSupervisor:
1. discover_temporal_columns() → Tìm time columns
2. execute_sql_query() → Lấy sample data
3. define_training_task() → Define prediction task
4. generate_sql_implementation() → Sinh SQL training table
5. generate_temporal_splits_from_db() → Create train/test splits
```

### 9.2 Graph Construction
```
RelationalGraphArchitect:
1. extract_schema_metadata() → Load DB schema
2. get_table_columns() → Verify column names (⚠️ snake_case!)
3. load_table_data() → Fetch tables
4. build_hetero_graph() → Xây graph từ relationships
5. encode_multi_modal_features() → Feature encoding
```

### 9.3 GNN Training
```
RelationalGNNSpecialist:
1. get_hetero_graph_from_registry() → Lấy graph
2. load_training_data() → Lấy labels
3. configure_temporal_sampler() → Time-aware sampling
4. build_gnn_model() → Xây PyTorch GNN
5. train_gnn_epoch() → Training loop
6. evaluate_gnn() → Evaluation
```

**Điểm quan trọng**:
- Snake_case columns: `owner_user_id` không phải `OwnerUserId`
- Temporal consistency: Không leak data từ future
- Graph structure: Multi-table relationships

---

## 10. DEBUG CHECKLIST

Để debug hệ thống cho relational database tasks:

### 10.1 Input Validation
- [ ] `intent` rõ ràng?
- [ ] `input_schema` định nghĩa features từ DB?
- [ ] `output_schema` định nghĩa target?
- [ ] `db_connection_string` đúng?

### 10.2 Temporal Task Phase
- [ ] `discover_temporal_columns()` tìm được time columns?
- [ ] `execute_sql_query()` chạy được SQL?
- [ ] `define_training_task()` define được task đúng?
- [ ] `generate_sql_implementation()` tạo được training table?

### 10.3 Graph Construction Phase
- [ ] `extract_schema_metadata()` parse được schema?
- [ ] **`get_table_columns()` trả về đúng snake_case names?** ⚠️
- [ ] `load_table_data()` fetch được data?
- [ ] `build_hetero_graph()` xây được graph?

### 10.4 GNN Training Phase
- [ ] Graph được register trong ObjectRegistry?
- [ ] Training data (labels) được load?
- [ ] GNN model được build?
- [ ] Training loop chạy được?

### 10.5 Code Generation
- [ ] Training code sinh được?
- [ ] Code validate pass?
- [ ] Code execute được trong isolated env?
- [ ] Inference code sinh được?

### 10.6 Evaluation
- [ ] Metrics computed đúng?
- [ ] Best solution selected đúng?
- [ ] Final output format đúng?

---

## 11. KEY FILES TO MONITOR

```
Core:
- plexe/models.py              # Model class
- plexe/model_builder.py       # ModelBuilder.build() entry
- plexe/agents/agents.py       # PlexeAgent orchestrator

Code Generation:
- plexe/agents/model_trainer.py           # Training code generation
- plexe/agents/model_packager.py          # Inference code generation
- plexe/tools/training.py                 # Training code tools
- plexe/internal/models/generation/       # Code generators

RDL (Database):
- plexe/agents/relational_graph_architect.py    # Graph construction
- plexe/agents/temporal_task_supervisor.py      # Temporal tasks
- plexe/agents/relational_gnn_specialist.py     # GNN training
- plexe/tools/graph_processing.py               # Graph tools
- plexe/tools/temporal_processing.py            # Temporal tools
- plexe/tools/gnn_processing.py                 # GNN tools

Execution:
- plexe/tools/execution.py                # Code execution
- plexe/internal/models/execution/        # Execution engines

Registry:
- plexe/core/object_registry.py           # Shared state
- plexe/core/entities/solution.py         # Solution object
```

---

## 12. EXECUTION FLOW DIAGRAM

```
ModelBuilder.build(
    intent="Predict X from DB",
    datasets=None,
    db_connection_string="postgresql://..."
)
    ↓
ObjectRegistry.register(db_connection_string)
    ↓
PlexeAgent.run(task, additional_args)
    ↓
Orchestrator (CodeAgent managed_agents)
    ├─→ MLResearchScientist (ToolCallingAgent)
    │   ├─ get_dataset_preview()
    │   ├─ get_solution_creation_tool()
    │   └─ → Solution ID + plan
    │
    ├─→ TemporalTaskSupervisor (CodeAgent)
    │   ├─ discover_temporal_columns()
    │   ├─ generate_sql_implementation()
    │   └─ → Training table SQL + splits
    │
    ├─→ RelationalGraphArchitect (CodeAgent)
    │   ├─ extract_schema_metadata()
    │   ├─ build_hetero_graph()
    │   └─ → HeteroData graph in registry
    │
    ├─→ DatasetSplitter (CodeAgent)
    │   └─ → Train/val splits
    │
    ├─→ FeatureEngineer (CodeAgent)
    │   ├─ Sinh feature transformation code
    │   └─ → Transformed datasets
    │
    ├─→ MLEngineer (CodeAgent) ⭐ MAIN
    │   ├─ generate_training_code() ← LLM sinh code
    │   ├─ validate_training_code()
    │   ├─ execute_training_code()
    │   ├─ Metrics computation
    │   └─ → Solution + training_code + performance
    │
    ├─→ RelationalGNNSpecialist (CodeAgent)
    │   ├─ build_gnn_model()
    │   ├─ train_gnn_epoch() × N
    │   ├─ evaluate_gnn()
    │   └─ → GNN model + metrics
    │
    ├─→ MLOperationsEngineer (CodeAgent)
    │   ├─ get_inference_context_tool()
    │   ├─ validate_inference_code()
    │   └─ → Solution + inference_code
    │
    ├─→ ModelTester (CodeAgent)
    │   ├─ execute test code
    │   ├─ generate_evaluation_report()
    │   └─ → Test performance + report
    │
    └─→ get_select_target_metric()
    └─→ register_best_solution()
    
    ↓
ModelGenerationResult {
    training_source_code: str,
    inference_source_code: str,
    predictor: Predictor,
    performance: Metric,
    evaluation_report: Dict,
}
```

---

## 13. COMMON DEBUG SCENARIOS

### Scenario 1: Generated Training Code Không Chạy

**Kiểm tra**:
1. Code có syntax errors? → Check `validate_training_code()` output
2. Datasets format đúng? → Check dataset schema
3. Imports đúng? → Check `additional_authorized_imports`
4. Timeout? → Increase timeout parameter
5. Check execution environment → Process executor logs

**Tools**:
- `plexe/tools/execution.py::execute_training_code()`
- `plexe/tools/validation.py::validate_training_code()`

---

### Scenario 2: RDL Code Sinh Ra Sai Column Names

**Kiểm tra**:
1. Column names snake_case? → Check DB schema
2. Agent gọi `get_table_columns()` không? → Force call
3. SQL query generate sai? → Check TemporalTaskSupervisor logs
4. Graph build sai? → Check RelationalGraphArchitect logs

**Tools**:
- `plexe/tools/temporal_processing.py::get_table_columns()`
- `plexe/tools/graph_processing.py::get_table_columns()`

---

### Scenario 3: Inference Code Không Load Model

**Kiểm tra**:
1. Model artifacts saved? → Check ObjectRegistry
2. PredictorImplementation class tạo được? → Check model_packager logs
3. Model artifact path đúng? → Check artifact registration
4. Pickle/torch load được? → Check serialization

---

## 14. QUICK START DEBUG

```bash
# 1. Enable verbose logging
export PLEXE_VERBOSE=1
export LOG_LEVEL=DEBUG

# 2. Run with minimal config
python -c "
from plexe.model_builder import ModelBuilder
import pandas as pd

df = pd.read_csv('sample.csv')
builder = ModelBuilder(verbose=True)
result = builder.build(
    intent='Predict X from features',
    datasets=[df],
)
print(f'Training code:\\n{result.training_source_code}')
print(f'Inference code:\\n{result.inference_source_code}')
print(f'Performance: {result.performance}')
"

# 3. Check ObjectRegistry
from plexe.core.object_registry import ObjectRegistry
reg = ObjectRegistry()
print(reg.list())

# 4. Check Solution objects
from plexe.core.entities.solution import Solution
solutions = reg.list_by_type(Solution)
for sol_id in solutions:
    sol = reg.get(Solution, sol_id)
    print(f'{sol_id}: training_code={len(sol.training_code)} chars')
```

---

## 15. METRICS & PERFORMANCE

Hệ thống track:
- **Training performance**: Metric trên validation set
- **Test performance**: Metric trên test set
- **Model size**: Model artifact size
- **Training time**: Execution time
- **Code quality**: Review scores

**File**: `plexe/internal/models/entities/metric.py`

```python
class Metric:
    name: str                    # e.g., "accuracy"
    value: float                 # e.g., 0.95
    comparator: MetricComparator # HIGHER_IS_BETTER / LOWER_IS_BETTER
```

---

## SUMMARY

**Hệ thống Plexe hoạt động như sau**:

1. **Input**: Intent + datasets/DB connection
2. **Planning**: MLResearchScientist → Solutions
3. **Preparation**: FeatureEngineer, DatasetSplitter → Processed data
4. **Training**: MLEngineer **sinh training code** + execute
5. **Inference**: MLOperationsEngineer **sinh inference code**
6. **Testing**: ModelTester → Evaluation report
7. **Selection**: Orchestrator selects best solution
8. **Output**: ModelGenerationResult (code + predictor + metrics)

**Agents sinh code**:
- ✓ MLEngineer (Training code)
- ✓ MLOperationsEngineer (Inference code)
- ✓ FeatureEngineer (Feature transformation)
- ✓ TemporalTaskSupervisor (SQL + temporal handling)
- ✓ RelationalGraphArchitect (Graph construction)
- ✓ RelationalGNNSpecialist (GNN training)

**Để debug relational database tasks**, focus on:
1. Temporal phase: Column discovery + SQL generation
2. Graph phase: Schema parsing + graph construction
3. GNN phase: Model building + training
4. Code generation phase: Code validation + execution
