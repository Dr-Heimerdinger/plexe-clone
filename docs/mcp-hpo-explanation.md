# MCP HPO Integration trong Plexe

## Tổng quan

Plexe sử dụng **MCP (Model Context Protocol)** để implement **Training-Free Hyperparameter Optimization (HPO)**. Thay vì training nhiều lần để tìm hyperparameters tối ưu, hệ thống truy vấn các nguồn kiến thức external (papers, benchmarks) để tìm cấu hình đã được proven.

## Kiến trúc MCP

```
┌─────────────────────────────────────────────────────────────┐
│                     GNN Specialist Agent                     │
│                   (needs hyperparameters)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       MCP Manager                            │
│           (converts MCP tools → LangChain tools)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    HPO MCP Server                            │
│              (hpo_server.py - FastMCP)                       │
├─────────────────────────────────────────────────────────────┤
│  Tools:                                                      │
│  ✓ search_optimal_hyperparameters()                         │
│  ✓ extract_hyperparameters_from_papers()                    │
│  ✓ get_benchmark_hyperparameters()                          │
│  ✓ compare_hyperparameter_configs()                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │  arXiv   │   │ Papers   │   │   Dataset    │
    │   API    │   │   With   │   │ Heuristics   │
    │          │   │   Code   │   │              │
    └──────────┘   └──────────┘   └──────────────┘
```

## Components

### 1. MCP Config ([mcp_config.json](../mcp_config.json))

```json
{
  "mcpServers": {
    "hpo-search": {
      "command": "python",
      "args": ["plexe/langgraph/mcp_servers/hpo_server.py"]
    }
  }
}
```

### 2. HPO MCP Server ([plexe/langgraph/mcp_servers/hpo_server.py](../plexe/langgraph/mcp_servers/hpo_server.py))

**4 MCP Tools chính:**

#### `search_optimal_hyperparameters()`
- **Input**: task_type, num_nodes, num_tables, is_temporal
- **Output**: Hyperparameters dựa trên heuristics và dataset scale
- **Logic**: Áp dụng rules từ literature (GraphSAGE, RelBench papers)
- **Ví dụ**:
  ```python
  {
    "hyperparameters": {
      "hidden_channels": 128,
      "batch_size": 512,
      "learning_rate": 0.008,  # 0.01 * 0.8 for temporal
      "num_gnn_layers": 2,
      "epochs": 15
    },
    "reasoning": "Based on 15000 nodes, temporal task...",
    "confidence": "high"
  }
  ```

#### `extract_hyperparameters_from_papers()`
- **Input**: paper_query, model_type, num_papers
- **Output**: Hyperparameters extracted từ papers thực
- **Logic**: 
  1. Query arXiv API
  2. Parse paper abstracts
  3. Extract hyperparams bằng regex (learning rate, batch size, etc.)
  4. Aggregate multiple papers
- **Ví dụ**:
  ```python
  {
    "papers_analyzed": 5,
    "papers_with_hyperparams": 3,
    "extracted_hyperparameters": [
      {
        "source_paper": "GraphSAGE: ...",
        "hyperparameters": {
          "learning_rate": 0.01,
          "batch_size": 512
        }
      }
    ],
    "aggregated_hyperparameters": {
      "learning_rate": 0.01,  # median từ 3 papers
      "batch_size": 512
    }
  }
  ```

#### `get_benchmark_hyperparameters()`
- **Input**: task_type, dataset_domain, model_architecture
- **Output**: Hyperparameters từ benchmark leaderboards
- **Logic**: Query Papers With Code API cho SOTA results
- **Ví dụ**:
  ```python
  {
    "hyperparameters": {
      "hidden_channels": 128,
      "num_gnn_layers": 2,
      "learning_rate": 0.005
    },
    "benchmarks_referenced": 2,
    "benchmark_papers": [
      {"paper_title": "...", "paper_url": "..."}
    ]
  }
  ```

#### `compare_hyperparameter_configs()`
- **Input**: List of configs from different sources, strategy
- **Output**: Ensemble recommendation
- **Logic**: Ensemble voting (median cho numeric, mode cho categorical)
- **Ví dụ**:
  ```python
  {
    "recommended_hyperparameters": {
      "learning_rate": 0.008,  # median of [0.01, 0.008, 0.005]
      "batch_size": 512,       # mode
      "hidden_channels": 128   # median of [128, 128, 256]
    },
    "num_sources": 3,
    "confidence": "high"
  }
  ```

### 3. GNN Specialist Agent ([plexe/langgraph/agents/gnn_specialist.py](../plexe/langgraph/agents/gnn_specialist.py))

**Workflow:**

1. **Agent init** → MCPManager loads HPO tools tự động từ BaseAgent
2. **Agent execution**:
   ```python
   # Step 1: Heuristic-based
   result_a = search_optimal_hyperparameters(
       task_type="binary_classification",
       num_nodes=15000,
       num_tables=7,
       is_temporal=True
   )
   
   # Step 2: Literature-based
   result_b = extract_hyperparameters_from_papers(
       paper_query="Relational GNN classification",
       model_type="gnn"
   )
   
   # Step 3: Benchmark-based
   result_c = get_benchmark_hyperparameters(
       task_type="binary_classification",
       dataset_domain="relational"
   )
   
   # Step 4: Ensemble
   final = compare_hyperparameter_configs(
       configs=[result_a, result_b, result_c]
   )
   
   # Step 5: Generate training script
   generate_training_script(
       ...,
       **final["recommended_hyperparameters"]
   )
   ```

3. **Output**: Training script với optimal hyperparameters + reasoning

## So sánh: Trước vs Sau

### ❌ Trước (Không dùng MCP đúng cách)

```
GNN Agent → tools/hpo_search.py → external_api_clients.py → Direct API calls
                                                            ↓
                                    Chỉ trả về paper metadata, không có hyperparams
```

**Vấn đề:**
- MCP servers (arxiv, semantic-scholar) chỉ return paper info
- HPO tools gọi trực tiếp API clients, không qua MCP
- Không có tool nào thực sự extract hyperparameters

### ✅ Sau (MCP integration đúng)

```
GNN Agent → MCP Manager → HPO MCP Server → arXiv API + Regex extraction
                                         ↓
                          Trả về hyperparameters cụ thể ready-to-use
```

**Cải thiện:**
- HPO MCP server dedicated cho hyperparameter search
- Extract hyperparams từ paper text (learning rate, batch size, etc.)
- Aggregate multiple sources
- Return format chuẩn cho training script

## Tác dụng

### 1. **Training-Free HPO**
Không cần chạy 10-20 training experiments để tune hyperparameters. Thay vào đó:
- Query 5 papers → extract their hyperparams → aggregate
- Tiết kiệm hàng giờ GPU time

### 2. **Knowledge-Based Optimization**
Leverage kiến thức từ:
- Hàng nghìn papers trên arXiv
- Benchmark leaderboards (Papers With Code)
- Proven heuristics từ literature

### 3. **Explainable Recommendations**
Mỗi hyperparameter có reasoning:
```
"Learning rate = 0.008 because:
 - Paper A used 0.01 for similar task
 - Paper B used 0.005
 - Median = 0.008
 - Temporal adjustment: -20%"
```

### 4. **Scalable Architecture**
Dễ dàng thêm MCP servers mới:
- OpenML server cho dataset benchmarks
- Kaggle server cho competition winners
- Custom servers cho internal knowledge base

## Testing

Run test để verify integration:

```bash
python test_mcp_hpo_integration.py
```

Expected output:
```
1. Initializing MCP Manager...
✓ MCP Manager initialized successfully
  Connected to 5 MCP servers

2. Checking loaded MCP tools...
✓ Loaded 12 total MCP tools
✓ Found 4 HPO-related tools:
  - search_optimal_hyperparameters
  - extract_hyperparameters_from_papers
  - get_benchmark_hyperparameters
  - compare_hyperparameter_configs

3. Testing HPO tools...
✓ All tools return hyperparameters successfully
```

## Future Enhancements

1. **LLM-based extraction**: Thay regex bằng LLM để extract hyperparams chính xác hơn
2. **Meta-learning**: Học từ past experiments để predict optimal hyperparams
3. **Active learning**: Query specific papers based on dataset characteristics
4. **Benchmark database**: Cache benchmark results để query nhanh hơn

## Kết luận

MCP trong Plexe không chỉ là infrastructure để kết nối external APIs. Nó là core component của **Training-Free HPO strategy**, cho phép:
- ✅ Tìm hyperparameters tối ưu WITHOUT expensive training
- ✅ Leverage collective knowledge từ research community
- ✅ Explainable và reproducible recommendations
- ✅ Extensible architecture cho future enhancements
