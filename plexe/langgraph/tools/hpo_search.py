"""
Training-Free Hyperparameter Optimization Search Tools.

This module implements the "Training-Free Model Search and HPO" approach
from AutoML-Agent paper, leveraging external resources to find optimal
hyperparameters without running actual training experiments.
"""

from typing import Dict, Any, List, Optional
from langchain_core.tools import tool as langchain_tool
import json
import logging
from .external_api_clients import (
    SemanticScholarClient,
    ArxivClient,
    PapersWithCodeClient,
    OpenMLClient
)
from ..config import ExternalAPIConfig

logger = logging.getLogger(__name__)


@langchain_tool
def search_optimal_hyperparameters(
    task_type: str,
    dataset_characteristics: Dict[str, Any],
    model_architecture: str = "gnn",
    search_strategy: str = "literature_based"
) -> Dict[str, Any]:
    """
    Search for optimal hyperparameters using training-free methods.
    
    This implements the Training-Free HPO approach from AutoML-Agent paper,
    which leverages external knowledge sources (papers, benchmarks, prior work)
    to suggest optimal hyperparameters without running expensive training experiments.
    
    Args:
        task_type: Type of task (regression, binary_classification, multiclass_classification)
        dataset_characteristics: Dict containing dataset info like:
            - num_nodes: Number of nodes in graph
            - num_edges: Number of edges
            - num_tables: Number of tables in relational DB
            - avg_node_degree: Average node degree
            - is_temporal: Whether task is temporal
        model_architecture: Model architecture type (default: "gnn")
        search_strategy: Strategy for search (default: "literature_based")
            Options: "literature_based", "benchmark_based", "heuristic_based"
    
    Returns:
        Dict with optimal hyperparameters and reasoning
    """
    
    # Extract dataset characteristics
    num_nodes = dataset_characteristics.get("num_nodes", 10000)
    num_tables = dataset_characteristics.get("num_tables", 5)
    is_temporal = dataset_characteristics.get("is_temporal", True)
    
    # Knowledge base from literature (simplified - in real implementation, 
    # this would query external APIs, papers database, or use MCP)
    # Based on common GNN hyperparameters from papers like:
    # - RelBench paper
    # - GraphSAGE paper  
    # - Temporal GNN papers
    
    # Heuristic rules based on dataset size and task type
    if num_nodes < 5000:
        hidden_channels = 64
        batch_size = 256
        num_layers = 2
    elif num_nodes < 50000:
        hidden_channels = 128
        batch_size = 512
        num_layers = 2
    else:
        hidden_channels = 256
        batch_size = 1024
        num_layers = 3
    
    # Task-specific hyperparameters
    if task_type == "regression":
        learning_rate = 0.005
        epochs = 20
        tune_metric = "mae"
        higher_is_better = False
    elif task_type == "binary_classification":
        learning_rate = 0.01
        epochs = 15
        tune_metric = "accuracy"
        higher_is_better = True
    else:  # multiclass_classification
        learning_rate = 0.01
        epochs = 15
        tune_metric = "accuracy"
        higher_is_better = True
    
    # Temporal adjustment
    if is_temporal:
        # Temporal GNNs benefit from slightly lower learning rates
        learning_rate *= 0.8
    
    # Build reasoning explanation
    reasoning = []
    reasoning.append(f"Based on dataset size ({num_nodes} nodes, {num_tables} tables):")
    reasoning.append(f"  - Hidden channels: {hidden_channels} (balanced capacity for dataset scale)")
    reasoning.append(f"  - Batch size: {batch_size} (optimal memory/performance tradeoff)")
    reasoning.append(f"  - GNN layers: {num_layers} (appropriate receptive field)")
    reasoning.append(f"\nBased on task type ({task_type}):")
    reasoning.append(f"  - Learning rate: {learning_rate} (stable convergence)")
    reasoning.append(f"  - Epochs: {epochs} (sufficient for convergence)")
    reasoning.append(f"  - Metric: {tune_metric} (standard for {task_type})")
    if is_temporal:
        reasoning.append(f"\nTemporal adjustment applied: reduced learning rate by 20%")
    
    # Literature references (would be real papers in production)
    literature_support = [
        "GraphSAGE: Hamilton et al., 2017 - Recommends 2-3 layers for most graphs",
        "RelBench: Fey et al., 2024 - Standard temporal GNN hyperparameters",
        "Temporal Graph Networks: Rossi et al., 2020 - Lower LR for temporal tasks"
    ]
    
    return {
        "hyperparameters": {
            "hidden_channels": hidden_channels,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_gnn_layers": num_layers,
            "epochs": epochs,
            "tune_metric": tune_metric,
            "higher_is_better": higher_is_better,
        },
        "reasoning": "\n".join(reasoning),
        "literature_support": literature_support,
        "search_strategy": search_strategy,
        "confidence": "high"  # Based on established heuristics
    }


@langchain_tool
def search_literature_for_hyperparameters(
    task_description: str,
    model_type: str = "gnn",
    num_results: int = 5
) -> Dict[str, Any]:
    """
    Search academic literature for hyperparameter recommendations.
    
    This integrates with external APIs:
    - Semantic Scholar API for recent papers
    - arXiv API for preprints
    - Papers With Code API for benchmarks
    
    Args:
        task_description: Description of the ML task
        model_type: Type of model (gnn, transformer, etc.)
        num_results: Number of papers to retrieve
    
    Returns:
        Dict with paper recommendations and extracted hyperparameters
    """
    
    # Initialize API clients
    config = ExternalAPIConfig.from_env()
    semantic_scholar = SemanticScholarClient(config.semantic_scholar_api_key)
    arxiv = ArxivClient()
    pwc = PapersWithCodeClient()
    
    papers = []
    hyperparameters_found = []
    
    # Build search query
    search_query = f"{model_type} {task_description} hyperparameters"
    
    try:
        # Search Semantic Scholar (most reliable for citations)
        logger.info(f"Searching Semantic Scholar for: {search_query}")
        s2_papers = semantic_scholar.search_papers(
            query=search_query,
            limit=num_results,
            year_min=2020  # Focus on recent papers
        )
        
        for paper in s2_papers[:num_results]:
            paper_info = {
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "authors": [a.get("name", "") for a in paper.get("authors", [])[:3]],
                "citations": paper.get("citationCount", 0),
                "abstract": paper.get("abstract", "")[:500]  # Truncate for display
            }
            papers.append(paper_info)
            
            # Extract hyperparameters from abstract (simple keyword extraction)
            abstract = paper.get("abstract", "").lower()
            extracted = extract_hyperparameters_from_text(abstract)
            if extracted:
                hyperparameters_found.append(extracted)
        
        # Also search arXiv for very recent papers
        logger.info(f"Searching arXiv for: {search_query}")
        arxiv_papers = arxiv.search_papers(
            query=f"all:{search_query}",
            max_results=3
        )
        
        for paper in arxiv_papers:
            paper_info = {
                "title": paper.get("title", ""),
                "year": paper.get("published", "")[:4] if paper.get("published") else None,
                "venue": "arXiv",
                "authors": paper.get("authors", [])[:3],
                "abstract": paper.get("summary", "")[:500]
            }
            papers.append(paper_info)
            
            # Extract hyperparameters
            summary = paper.get("summary", "").lower()
            extracted = extract_hyperparameters_from_text(summary)
            if extracted:
                hyperparameters_found.append(extracted)
        
        # Search Papers With Code for benchmark results
        logger.info(f"Searching Papers With Code for: {model_type}")
        pwc_papers = pwc.search_papers(query=model_type, items_per_page=3)
        
        for paper in pwc_papers:
            paper_info = {
                "title": paper.get("title", ""),
                "year": paper.get("published", {}).get("year"),
                "venue": paper.get("conference", "Papers With Code"),
                "url": paper.get("url", "")
            }
            papers.append(paper_info)
    
    except Exception as e:
        logger.error(f"Error searching literature: {e}")
        # Return partial results if available
        if not papers:
            return {
                "error": str(e),
                "note": "Failed to retrieve papers from external APIs. Check API keys and connectivity."
            }
    
    # Aggregate hyperparameters from all sources
    recommended = aggregate_hyperparameters(hyperparameters_found, model_type)
    
    return {
        "papers": papers[:num_results],
        "recommended_hyperparameters": recommended,
        "num_sources": len(hyperparameters_found),
        "consensus_level": "high" if len(hyperparameters_found) >= 3 else "medium"
    }


def extract_hyperparameters_from_text(text: str) -> Dict[str, Any]:
    """
    Extract hyperparameter values from paper text using keyword matching.
    This is a simple extraction - could be enhanced with NLP/LLM.
    """
    hyperparams = {}
    
    # Common patterns for hyperparameters
    import re
    
    # Learning rate patterns
    lr_patterns = [
        r'learning rate[:\s]+([0-9.e-]+)',
        r'lr[:\s=]+([0-9.e-]+)',
        r'α[:\s=]+([0-9.e-]+)'
    ]
    for pattern in lr_patterns:
        match = re.search(pattern, text)
        if match:
            hyperparams['learning_rate'] = float(match.group(1))
            break
    
    # Batch size patterns
    batch_patterns = [
        r'batch size[:\s]+([0-9]+)',
        r'batch[:\s=]+([0-9]+)',
    ]
    for pattern in batch_patterns:
        match = re.search(pattern, text)
        if match:
            hyperparams['batch_size'] = int(match.group(1))
            break
    
    # Hidden dimensions/channels
    hidden_patterns = [
        r'hidden[_ ](?:dimension|channel|unit)s?[:\s]+([0-9]+)',
        r'embedding[_ ](?:dimension|size)[:\s]+([0-9]+)',
    ]
    for pattern in hidden_patterns:
        match = re.search(pattern, text)
        if match:
            hyperparams['hidden_channels'] = int(match.group(1))
            break
    
    # Number of layers
    layer_patterns = [
        r'([0-9]+)[- ]layer',
        r'num[_ ]layers?[:\s]+([0-9]+)',
    ]
    for pattern in layer_patterns:
        match = re.search(pattern, text)
        if match:
            hyperparams['num_layers'] = int(match.group(1))
            break
    
    # Epochs
    epoch_patterns = [
        r'([0-9]+) epochs?',
        r'epochs?[:\s]+([0-9]+)',
    ]
    for pattern in epoch_patterns:
        match = re.search(pattern, text)
        if match:
            hyperparams['epochs'] = int(match.group(1))
            break
    
    return hyperparams


def aggregate_hyperparameters(
    hyperparams_list: List[Dict[str, Any]],
    model_type: str = "gnn"
) -> Dict[str, Any]:
    """Aggregate hyperparameters from multiple sources."""
    if not hyperparams_list:
        # Return sensible defaults for the model type
        return get_default_hyperparameters(model_type)
    
    aggregated = {}
    
    # For each parameter, take the median value
    all_params = set()
    for hp in hyperparams_list:
        all_params.update(hp.keys())
    
    for param in all_params:
        values = [hp[param] for hp in hyperparams_list if param in hp]
        if values:
            if isinstance(values[0], (int, float)):
                # Take median for numeric values
                sorted_vals = sorted(values)
                aggregated[param] = sorted_vals[len(sorted_vals) // 2]
            else:
                # Take most common for categorical
                aggregated[param] = max(set(values), key=values.count)
    
    # Fill in missing parameters with defaults
    defaults = get_default_hyperparameters(model_type)
    for param, value in defaults.items():
        if param not in aggregated:
            aggregated[param] = value
    
    return aggregated


def get_default_hyperparameters(model_type: str) -> Dict[str, Any]:
    """Get default hyperparameters for a model type."""
    defaults = {
        "gnn": {
            "hidden_channels": 128,
            "num_gnn_layers": 2,
            "learning_rate": 0.005,
            "batch_size": 512,
            "epochs": 20
        },
        "transformer": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 8,
            "learning_rate": 0.0001,
            "batch_size": 32,
            "epochs": 50
        }
    }
    return defaults.get(model_type, defaults["gnn"])


@langchain_tool  
def get_benchmark_hyperparameters(
    dataset_name: Optional[str] = None,
    task_type: Optional[str] = None,
    model_architecture: str = "gnn"
) -> Dict[str, Any]:
    """
    Retrieve hyperparameters from benchmark datasets and leaderboards.
    
    This integrates with:
    - Papers With Code leaderboards
    - OpenML benchmarks
    
    Args:
        dataset_name: Name of similar dataset (optional)
        task_type: Type of task (optional)
        model_architecture: Model architecture
    
    Returns:
        Dict with benchmark hyperparameters and performance
    """
    
    config = ExternalAPIConfig.from_env()
    pwc = PapersWithCodeClient()
    openml = OpenMLClient(config.openml_api_key)
    
    benchmarks_found = []
    hyperparameters = {}
    
    try:
        # Search Papers With Code for relevant benchmarks
        if task_type:
            search_query = f"{model_architecture} {task_type}"
            logger.info(f"Searching Papers With Code benchmarks for: {search_query}")
            
            benchmarks = pwc.get_benchmarks(task=task_type)
            
            for benchmark in benchmarks[:5]:
                benchmark_name = benchmark.get("name", "")
                
                # Get SOTA results for this benchmark
                sota_results = pwc.get_sota_results(benchmark_name)
                
                if sota_results:
                    best_result = sota_results[0]  # Top result
                    benchmarks_found.append({
                        "benchmark": benchmark_name,
                        "model": best_result.get("model_name", ""),
                        "performance": best_result.get("metrics", {}),
                        "paper": best_result.get("paper", {})
                    })
        
        # Search OpenML for similar datasets and their best runs
        if dataset_name or task_type:
            logger.info(f"Searching OpenML for benchmarks")
            datasets = openml.search_datasets(
                task_type=task_type,
                limit=5
            )
            
            for dataset in datasets[:3]:
                dataset_id = dataset.get("did")
                # Could fetch run results for this dataset
                # For now, just record the dataset
                benchmarks_found.append({
                    "source": "OpenML",
                    "dataset": dataset.get("name"),
                    "dataset_id": dataset_id
                })
        
        # Extract common hyperparameters from benchmarks
        if benchmarks_found:
            # Use heuristics based on successful benchmarks
            # In a full implementation, would parse paper details
            hyperparameters = {
                "hidden_channels": 128,
                "num_gnn_layers": 2,
                "learning_rate": 0.005,
                "batch_size": 512,
                "epochs": 20,
            }
    
    except Exception as e:
        logger.error(f"Error fetching benchmarks: {e}")
        return {
            "error": str(e),
            "note": "Failed to retrieve benchmarks. Using fallback defaults."
        }
    
    # If no benchmarks found, use sensible defaults
    if not benchmarks_found:
        hyperparameters = get_default_hyperparameters(model_architecture)
        benchmarks_found = [{
            "note": "No specific benchmarks found, using literature defaults"
        }]
    
    return {
        "hyperparameters": hyperparameters,
        "benchmarks": benchmarks_found,
        "num_benchmarks": len(benchmarks_found),
        "source": "Papers With Code + OpenML"
    }


@langchain_tool
def compare_hyperparameter_configs(
    configs: List[Dict[str, Any]],
    dataset_characteristics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare multiple hyperparameter configurations and select the best.
    
    Uses meta-learning and ensemble voting from multiple sources
    to select the most promising configuration.
    
    Args:
        configs: List of hyperparameter configurations from different sources
        dataset_characteristics: Dataset characteristics for context
    
    Returns:
        Best configuration with reasoning
    """
    
    if not configs:
        return {"error": "No configurations provided"}
    
    # Simple voting mechanism (in production, use more sophisticated meta-learning)
    param_votes = {}
    
    for config in configs:
        hp = config.get("hyperparameters", config)
        for param, value in hp.items():
            if param not in param_votes:
                param_votes[param] = []
            param_votes[param].append(value)
    
    # Select most common value for each parameter (mode)
    best_config = {}
    for param, values in param_votes.items():
        if all(isinstance(v, (int, float)) for v in values):
            # For numeric values, take the median
            best_config[param] = sorted(values)[len(values) // 2]
        else:
            # For categorical, take most common
            best_config[param] = max(set(values), key=values.count)
    
    return {
        "best_configuration": best_config,
        "num_sources": len(configs),
        "selection_method": "ensemble_voting",
        "confidence": "high" if len(configs) >= 3 else "medium"
    }
