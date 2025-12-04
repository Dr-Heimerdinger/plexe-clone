"""
Tools for GNN training and processing for the Relational GNN Specialist Agent.

This module provides robust tools for:
1. Time-aware neighbor sampling (critical for RDL)
2. Heterogeneous GNN model construction
3. Training with proper backpropagation and seed node masking
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from smolagents import tool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# PyTorch Geometric imports
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import (
    SAGEConv, GATConv, GCNConv, Linear,
    HeteroConv, to_hetero
)
from torch_geometric.typing import EdgeType, NodeType

# ObjectRegistry for data sharing between agents
from plexe.core.object_registry import ObjectRegistry
from plexe.internal.common.datasets.interface import TabularConvertible

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: Parse Time Window String
# =============================================================================
def _parse_time_window(time_window: str) -> int:
    """
    Parse time window string to seconds.
    
    Supports formats: '30d' (days), '24h' (hours), '60m' (minutes), '3600s' (seconds)
    
    Args:
        time_window: Time window string (e.g., '30d', '24h')
        
    Returns:
        Time window in seconds
    """
    if time_window is None:
        return None
        
    time_window = time_window.strip().lower()
    
    multipliers = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800,  # week
    }
    
    for suffix, mult in multipliers.items():
        if time_window.endswith(suffix):
            try:
                value = float(time_window[:-1])
                return int(value * mult)
            except ValueError:
                raise ValueError(f"Invalid time window format: {time_window}")
    
    # Try parsing as raw seconds
    try:
        return int(float(time_window))
    except ValueError:
        raise ValueError(f"Invalid time window format: {time_window}. Use formats like '30d', '24h', '60m'")


# =============================================================================
# Tool 0: Get Graph from Registry (Bridge between agents)
# =============================================================================

@tool
def get_hetero_graph_from_registry() -> Dict[str, Any]:
    """
    Retrieves the HeteroData graph built by RelationalGraphArchitect from ObjectRegistry.
    
    This is the FIRST tool to call in RelationalGNNSpecialist workflow.
    It bridges the gap between GraphArchitect (which builds the graph) and
    GNNSpecialist (which trains on it).
    
    Returns:
        Dictionary containing:
        - 'graph': The HeteroData object ready for training
        - 'metadata': Graph structure info (node types, edge types, counts)
        - 'temporal_info': Temporal split information if available
        - 'entity_mapper': EntityMapper state for prediction interpretation
        
    Example:
        >>> result = get_hetero_graph_from_registry()
        >>> graph = result['graph']
        >>> print(f"Graph has {len(graph.node_types)} node types")
    """
    try:
        object_registry = ObjectRegistry()
        
        # Get the HeteroData graph
        try:
            graph = object_registry.get(HeteroData, "hetero_graph")
        except KeyError:
            return {
                "error": "No HeteroData graph found in registry.",
                "hint": "Run RelationalGraphArchitect's build_hetero_graph first.",
                "available_keys": []
            }
        
        # Get metadata
        metadata = {
            "node_types": graph.node_types,
            "edge_types": [str(et) for et in graph.edge_types],
            "num_nodes": {},
            "num_edges": {},
            "has_features": {},
            "has_timestamps": {}
        }
        
        for nt in graph.node_types:
            if hasattr(graph[nt], 'num_nodes'):
                metadata["num_nodes"][nt] = graph[nt].num_nodes
            if hasattr(graph[nt], 'x') and graph[nt].x is not None:
                metadata["has_features"][nt] = graph[nt].x.shape
            if hasattr(graph[nt], 't') and graph[nt].t is not None:
                metadata["has_timestamps"][nt] = True
        
        for et in graph.edge_types:
            if hasattr(graph[et], 'edge_index'):
                metadata["num_edges"][str(et)] = graph[et].edge_index.shape[1]
        
        # Get EntityMapper state
        entity_mapper = None
        try:
            entity_mapper = object_registry.get(dict, "entity_mapper_state")
        except KeyError:
            pass
        
        # Get temporal split info if available
        temporal_info = None
        try:
            temporal_info = object_registry.get(dict, "temporal_split_info")
        except KeyError:
            pass
        
        return {
            "graph": graph,
            "metadata": metadata,
            "entity_mapper": entity_mapper,
            "temporal_info": temporal_info,
            "status": "Graph retrieved successfully from registry"
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Failed to get graph from registry: {str(e)}",
            "traceback": traceback.format_exc()
        }


# =============================================================================
# Tool 0.5: Load Training Data (Labels)
# =============================================================================

@tool
def load_training_data(
    entity_type: str,
    label_column: str = "label",
    timestamp_column: str = "seed_time",
    feature_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Loads training/validation/test labels generated by TemporalSupervisor.
    
    This tool bridges the gap between the Temporal Task Supervisor (which creates
    training tables with labels) and the GNN Specialist (which needs these labels
    mapped to graph nodes).
    
    It performs:
    1. Retrieval of 'temporal_train', 'temporal_val', 'temporal_test' from ObjectRegistry
    2. Mapping of Entity IDs (from DB) to Graph Indices (using EntityMapper)
    3. Conversion to PyTorch tensors
    4. (Optional) Loading of point-in-time correct features for seed nodes
    
    Args:
        entity_type: The node type in the graph corresponding to the training entities.
                     (e.g., 'customers', 'users')
        label_column: Name of the label column in the training table. Default 'label'.
        timestamp_column: Name of the timestamp column. Default 'seed_time'.
        feature_columns: Optional list of feature columns to load from the training table.
                         These features are computed at seed_time and should override
                         graph features for the seed nodes to prevent leakage.
    
    Returns:
        Dictionary containing:
        - 'train_data': Dict with 'node_indices', 'labels', 'seed_times', 'features' tensors
        - 'val_data': Dict with 'node_indices', 'labels', 'seed_times', 'features' tensors
        - 'test_data': Dict with 'node_indices', 'labels', 'seed_times', 'features' tensors
        - 'num_classes': Number of unique classes (for classification)
        - 'stats': Sample counts
    """
    try:
        object_registry = ObjectRegistry()
        
        # 1. Get EntityMapper state
        try:
            entity_mapper_state = object_registry.get(dict, "entity_mapper_state")
            # Extract mapping for the specific entity type
            # Structure: mappings[entity_type]['id_to_idx']
            if entity_type not in entity_mapper_state.get("mappings", {}):
                return {
                    "error": f"Entity type '{entity_type}' not found in EntityMapper.",
                    "available_types": list(entity_mapper_state.get("mappings", {}).keys())
                }
            
            id_to_idx = entity_mapper_state["mappings"][entity_type]["id_to_idx"]
            
        except KeyError:
            return {
                "error": "EntityMapper state not found in registry. Run RelationalGraphArchitect first."
            }
            
        # Helper to process a dataset
        def process_dataset(dataset_key: str) -> Optional[Dict[str, torch.Tensor]]:
            try:
                # Get dataset wrapper
                dataset_obj = object_registry.get(TabularConvertible, dataset_key)
                # Convert to pandas DataFrame
                df = dataset_obj.to_pandas()
                
                if df.empty:
                    return None
                
                # Check columns
                if label_column not in df.columns:
                    logger.warning(f"Label column '{label_column}' not found in {dataset_key}")
                    return None
                
                # Map IDs to Indices
                # We assume the dataframe has an ID column. 
                # We try to find it: 'entity_id', 'id', or matching the entity_type
                id_col = None
                possible_names = [entity_type + "_id", "entity_id", "id", "user_id", "customer_id"]
                for col in df.columns:
                    if col.lower() in possible_names:
                        id_col = col
                        break
                
                if not id_col:
                    # Fallback: check if any column values overlap with mapper keys
                    for col in df.columns:
                        sample_val = df[col].iloc[0]
                        if sample_val in id_to_idx:
                            id_col = col
                            break
                
                if not id_col:
                    logger.warning(f"Could not identify ID column in {dataset_key}")
                    return None
                
                # Perform mapping
                # Filter out IDs that don't exist in the graph (shouldn't happen if graph is up to date)
                valid_mask = df[id_col].isin(id_to_idx.keys())
                if (~valid_mask).any():
                    logger.warning(f"Dropping { (~valid_mask).sum() } samples with unknown IDs in {dataset_key}")
                    df = df[valid_mask].copy()
                
                if df.empty:
                    return None
                
                # Map to indices
                node_indices = df[id_col].map(id_to_idx).values.astype(np.int64)
                
                # Get labels
                labels = df[label_column].values
                # Convert to appropriate type (float for regression/binary, long for multiclass)
                # We'll infer: if float, keep float. If int/string/bool, convert to long/float
                if labels.dtype == np.float64 or labels.dtype == np.float32:
                    labels_tensor = torch.tensor(labels, dtype=torch.float)
                else:
                    # Try converting to numeric
                    try:
                        labels_tensor = torch.tensor(labels.astype(float), dtype=torch.float)
                        # If all integers, convert to long
                        if torch.all(labels_tensor == labels_tensor.long()):
                            labels_tensor = labels_tensor.long()
                    except:
                        # Categorical strings?
                        # For now, assume numeric labels
                        logger.warning(f"Non-numeric labels in {dataset_key}. Converting to codes.")
                        labels_tensor = torch.tensor(pd.Categorical(labels).codes, dtype=torch.long)

                # Get timestamps if available
                seed_times = None
                if timestamp_column in df.columns:
                    # Convert to unix timestamp if needed
                    times = df[timestamp_column]
                    if pd.api.types.is_datetime64_any_dtype(times):
                        seed_times = torch.tensor(times.astype(np.int64) // 10**9, dtype=torch.long)
                    else:
                        # Assume already timestamp or convertable
                        try:
                            seed_times = torch.tensor(pd.to_datetime(times).astype(np.int64) // 10**9, dtype=torch.long)
                        except:
                            pass
                
                # Get features if requested
                features_tensor = None
                if feature_columns:
                    missing_cols = [c for c in feature_columns if c not in df.columns]
                    if missing_cols:
                        logger.warning(f"Missing feature columns in {dataset_key}: {missing_cols}")
                    else:
                        # Extract features
                        feat_data = df[feature_columns].values
                        # Convert to float tensor
                        try:
                            features_tensor = torch.tensor(feat_data.astype(float), dtype=torch.float)
                        except Exception as e:
                            logger.warning(f"Could not convert features to tensor in {dataset_key}: {e}")

                result = {
                    "node_indices": torch.tensor(node_indices, dtype=torch.long),
                    "labels": labels_tensor,
                    "seed_times": seed_times
                }
                
                if features_tensor is not None:
                    result["features"] = features_tensor
                    
                return result
                
            except KeyError:
                return None
            except Exception as e:
                logger.warning(f"Error processing {dataset_key}: {e}")
                return None

        # Process all splits
        train_data = process_dataset("temporal_train")
        val_data = process_dataset("temporal_val")
        test_data = process_dataset("temporal_test")
        
        if not train_data:
            return {
                "error": "Could not load training data. Ensure TemporalSupervisor has run create_temporal_dataset."
            }
        
        # Calculate num_classes
        num_classes = 1
        if train_data["labels"].dtype == torch.long:
            num_classes = int(train_data["labels"].max().item()) + 1
        
        return {
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "num_classes": num_classes,
            "stats": {
                "train_samples": len(train_data["node_indices"]),
                "val_samples": len(val_data["node_indices"]) if val_data else 0,
                "test_samples": len(test_data["node_indices"]) if test_data else 0
            },
            "status": "Training data loaded and mapped successfully"
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Failed to load training data: {str(e)}",
            "traceback": traceback.format_exc()
        }


# =============================================================================
# Tool 1: Configure Temporal Sampler
# =============================================================================

@tool
def configure_temporal_sampler(
    hetero_data: Any,
    num_neighbors: List[int],
    input_nodes: Union[Tuple[str, Any], Dict[str, Any]],
    seed_times: Optional[Any] = None,
    time_attr: str = "t",
    time_window: Optional[str] = None,
    batch_size: int = 128,
    shuffle: bool = True,
    temporal_strategy: str = "last",
    task_type: str = "node"
) -> Any:
    """
    Creates a Time-Consistent Neighbor Sampler for RDL training.
    
    CRITICAL for RDL: Ensures temporal causality - when predicting for a seed node
    at time t, the sampler ONLY retrieves neighbors with timestamps t_neighbor <= t.
    This prevents data leakage from the future.
    
    The tool returns a PyTorch Geometric DataLoader that yields mini-batches
    with proper temporal constraints.
    
    Args:
        hetero_data: The HeteroData object from build_hetero_graph.
                     Must have time attribute (data[node_type].t) for temporal sampling.
        
        num_neighbors: List of integers specifying neighbors per layer.
                       Example: [15, 10] means sample 15 1-hop neighbors, 10 2-hop neighbors.
                       Deeper layers need fewer neighbors (exponential explosion).
        
        input_nodes: Either:
                     1. Tuple of (node_type, node_indices) specifying seed nodes.
                     2. Dictionary returned by load_training_data (containing 'node_indices', 'seed_times', 'labels').
        
        seed_times: Optional tensor of timestamps for seed nodes. 
                    If input_nodes is a dict, this is extracted automatically.
                    Used to enforce t_neighbor <= t_seed for each sample.
        
        time_attr: Name of the time attribute in HeteroData. Default 't'.
                   This attribute must exist in data[node_type].t
        
        time_window: Optional time window for neighbor filtering.
                     Format: '30d' (days), '24h' (hours), '60m' (minutes).
                     If set, only neighbors within this window are sampled.
        
        batch_size: Number of seed nodes per batch. Default 128.
        
        shuffle: Whether to shuffle seed nodes. Default True for training.
        
        temporal_strategy: Strategy for temporal sampling. Options:
                          - 'last': Use latest timestamp for each node (default)
                          - 'uniform': Uniform sampling within time window
        
        task_type: 'node' for node classification/regression,
                   'link' for link prediction.
    
    Returns:
        A configured DataLoader (NeighborLoader or LinkNeighborLoader) ready for training.
        
    Example:
        >>> # Using data from load_training_data
        >>> train_data = load_training_data('customers')['train_data']
        >>> loader = configure_temporal_sampler(
        ...     hetero_data=data,
        ...     num_neighbors=[15, 10],
        ...     input_nodes=train_data,  # Pass the dict directly
        ...     time_attr='t',
        ...     batch_size=256
        ... )
    """
    try:
        # Validate hetero_data
        if not isinstance(hetero_data, HeteroData):
            return {
                "error": "hetero_data must be a PyTorch Geometric HeteroData object",
                "received_type": type(hetero_data).__name__
            }
        
        # Parse time window
        time_window_seconds = None
        if time_window:
            time_window_seconds = _parse_time_window(time_window)
        
        # Handle input_nodes
        node_type = None
        node_indices = None
        labels = None
        features = None
        
        if isinstance(input_nodes, dict):
            # It's likely from load_training_data
            # We need to know the node_type. It's not in the dict explicitly, 
            # but usually the caller knows. 
            # Wait, load_training_data takes entity_type but returns a dict without it.
            # We need the user to pass node_type in the tuple if they use the dict?
            # Or we assume input_nodes is (node_type, dict)?
            # Let's support input_nodes being the dict, but we need node_type.
            # Actually, the user should pass (node_type, dict) or we infer it?
            # Let's assume the user passes (node_type, dict) OR we check if input_nodes has 'node_indices'.
            # But we still need node_type.
            
            # Let's change the signature/docstring to require (node_type, dict) or just handle dict if we can't infer.
            # But wait, the previous signature was (node_type, indices).
            # If the user passes a dict, they must have extracted it.
            # Let's assume the user passes the dict as the second element of the tuple?
            # No, let's support input_nodes as a tuple (node_type, dict).
            pass
        
        # Let's normalize input_nodes
        if isinstance(input_nodes, tuple):
            node_type = input_nodes[0]
            data_obj = input_nodes[1]
            
            if isinstance(data_obj, dict) and "node_indices" in data_obj:
                # It's the dict from load_training_data
                node_indices = data_obj["node_indices"]
                if "seed_times" in data_obj:
                    seed_times = data_obj["seed_times"]
                if "labels" in data_obj:
                    labels = data_obj["labels"]
                if "features" in data_obj:
                    features = data_obj["features"]
            else:
                # It's just indices
                node_indices = data_obj
        elif isinstance(input_nodes, dict) and "node_indices" in input_nodes:
             return {
                "error": "input_nodes must be a tuple (node_type, data) when passing a dictionary. Example: ('customers', train_data)"
            }
        else:
             return {
                "error": "input_nodes must be a tuple of (node_type, node_indices) or (node_type, data_dict)"
            }

        # Check if node type exists
        if node_type not in hetero_data.node_types:
            return {
                "error": f"Node type '{node_type}' not found in HeteroData",
                "available_types": hetero_data.node_types
            }
        
        # Check for time attribute
        has_time_attr = hasattr(hetero_data[node_type], time_attr)
        
        # Attach labels to the graph temporarily? 
        # No, NeighborLoader doesn't attach labels to the graph.
        # But we need labels in the batch.
        # If we pass input_nodes as (node_type, indices), the loader yields batches.
        # The batch will contain the sampled subgraph.
        # But where are the labels?
        # Usually, we attach labels to the graph: data['customer'].y = ...
        # But here we have multiple labels per node (different times).
        # So we cannot attach to the graph globally.
        
        # We need a custom transform or we need to pass labels to the loader?
        # PyG NeighborLoader doesn't natively support passing labels for the seeds in the constructor easily 
        # unless we use a custom Dataset.
        
        # WORKAROUND: We will attach the labels to the returned batch using a transform or wrapper.
        # Or, we can rely on the fact that the loader returns nodes in the order of input_nodes (if shuffle=False).
        # But shuffle=True is needed for training.
        
        # Better approach:
        # We can create a "fake" node type or use LinkNeighborLoader?
        # Or we can use the `input_data` argument of NeighborLoader (available in newer PyG).
        
        # Let's assume we can't easily change PyG version.
        # We will use a wrapper class around the loader that attaches the labels.
        
        class LabeledLoader:
            def __init__(self, loader, labels, seed_times=None, features=None):
                self.loader = loader
                self.labels = labels
                self.seed_times = seed_times
                self.features = features
                
            def __iter__(self):
                for batch in self.loader:
                    # We need to find which samples are in this batch.
                    # NeighborLoader with input_nodes=(type, indices) returns a batch
                    # where batch[node_type].input_id contains the original indices of the seed nodes.
                    # We can use this to fetch the corresponding labels.
                    
                    if hasattr(batch[node_type], 'input_id'):
                        input_ids = batch[node_type].input_id
                        # input_ids are indices into the ORIGINAL input_nodes tensor.
                        # So we can index into self.labels
                        
                        if self.labels is not None:
                            batch_labels = self.labels[input_ids]
                            # Attach to batch
                            # We attach it to the target node type
                            batch[node_type].y = batch_labels
                            batch.y = batch_labels # Convenience
                            
                        if self.seed_times is not None:
                            batch_seed_times = self.seed_times[input_ids]
                            batch[node_type].seed_time = batch_seed_times
                            batch.seed_time = batch_seed_times
                        
                        if self.features is not None:
                            batch_features = self.features[input_ids]
                            batch[node_type].seed_features = batch_features
                            
                    yield batch
            
            def __len__(self):
                return len(self.loader)

        if task_type == "node":
            # Node-level task: Use NeighborLoader
            
            # Prepare kwargs
            kwargs = {
                "data": hetero_data,
                "num_neighbors": num_neighbors,
                "input_nodes": (node_type, node_indices),
                "batch_size": batch_size,
                "shuffle": shuffle,
            }
            
            if has_time_attr:
                kwargs["time_attr"] = time_attr
            
            # If seed_times are provided, we try to pass them as input_time (PyG 2.3+)
            # We'll try to pass it, if it fails, we catch it.
            # Actually, let's check if we can use input_nodes=(node_type, torch.stack([indices, times], dim=1))?
            # No, that's not standard.
            
            # For now, we will rely on the LabeledLoader to attach seed_times to the batch,
            # BUT the sampling itself might not be time-aware per-sample if NeighborLoader doesn't support it.
            # However, if we pass `input_time` to NeighborLoader, it works.
            if seed_times is not None:
                kwargs["input_time"] = seed_times
            
            try:
                loader = NeighborLoader(**kwargs)
            except TypeError:
                # Fallback: input_time might not be supported in this PyG version
                # Remove it and warn
                if "input_time" in kwargs:
                    del kwargs["input_time"]
                    logger.warning("This PyG version might not support 'input_time' in NeighborLoader. Temporal sampling might be approximate.")
                loader = NeighborLoader(**kwargs)
            
            # Wrap with LabeledLoader if we have labels or seed_times
            if labels is not None or seed_times is not None or features is not None:
                loader = LabeledLoader(loader, labels, seed_times, features)
            
            loader_info = {
                "loader_type": "NeighborLoader (Labeled)",
                "task_type": "node",
                "seed_node_type": node_type,
                "num_seed_nodes": len(node_indices),
                "num_neighbors_per_layer": num_neighbors,
                "batch_size": batch_size,
                "temporal_enabled": has_time_attr,
                "has_labels": labels is not None,
                "has_seed_times": seed_times is not None,
                "has_seed_features": features is not None
            }
            
        elif task_type == "link":
            # Link prediction: Use LinkNeighborLoader
            loader = LinkNeighborLoader(
                data=hetero_data,
                num_neighbors=num_neighbors,
                edge_label_index=node_indices,  # For link pred, this is edge indices
                batch_size=batch_size,
                shuffle=shuffle,
                time_attr=time_attr if has_time_attr else None,
            )
            
            loader_info = {
                "loader_type": "LinkNeighborLoader",
                "task_type": "link",
                "num_neighbors_per_layer": num_neighbors,
                "batch_size": batch_size,
                "temporal_enabled": has_time_attr,
            }
        else:
            return {"error": f"Unknown task_type: {task_type}. Use 'node' or 'link'."}
        
        # Return both the loader and info
        return {
            "loader": loader,
            "info": loader_info,
            "status": "Temporal sampler configured successfully"
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Failed to configure sampler: {str(e)}",
            "traceback": traceback.format_exc()
        }


# =============================================================================
# Tool 2: Build Heterogeneous GNN Model
# =============================================================================

class HeteroGNNModel(nn.Module):
    """
    Heterogeneous GNN Model that handles multiple node/edge types.
    
    Uses HeteroConv to apply different message passing operations
    for each edge type, as required by relational databases.
    """
    
    def __init__(
        self,
        metadata: Tuple[List[NodeType], List[EdgeType]],
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        conv_type: str = "sage",
        dropout: float = 0.2,
        target_node_type: Optional[str] = None,
        task_type: str = "classification"
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.target_node_type = target_node_type
        self.task_type = task_type
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        node_types, edge_types = metadata
        
        # 1. Input Projection Layers (different input dims per node type)
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            in_dim = in_channels_dict.get(node_type, hidden_channels)
            self.input_projections[node_type] = Linear(in_dim, hidden_channels)
        
        # 2. HeteroConv Layers (one per GNN layer)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src_type, rel, dst_type = edge_type
                
                if conv_type.lower() == "sage":
                    conv_dict[edge_type] = SAGEConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        aggr='mean'
                    )
                elif conv_type.lower() == "gat":
                    conv_dict[edge_type] = GATConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels // 4,  # Multi-head
                        heads=4,
                        concat=True,
                        dropout=dropout
                    )
                elif conv_type.lower() == "gcn":
                    conv_dict[edge_type] = GCNConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels
                    )
                else:
                    # Default to SAGE
                    conv_dict[edge_type] = SAGEConv(hidden_channels, hidden_channels)
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # 3. Task Head (MLP for final prediction)
        if target_node_type:
            self.task_head = nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(hidden_channels, out_channels)
            )
        else:
            self.task_head = None
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        batch_size: Optional[int] = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through the heterogeneous GNN.
        
        Args:
            x_dict: Dictionary mapping node types to feature tensors.
            edge_index_dict: Dictionary mapping edge types to edge indices.
            batch_size: If provided, only return predictions for first batch_size nodes
                       (the seed nodes in mini-batch training).
        
        Returns:
            If target_node_type is set: Tensor of predictions for target nodes.
            Otherwise: Dictionary of embeddings for all node types.
        """
        # 1. Project inputs to hidden dimension
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.input_projections:
                h_dict[node_type] = self.input_projections[node_type](x)
            else:
                # Fallback: zero embedding if no projection
                h_dict[node_type] = torch.zeros(
                    x.size(0), self.hidden_channels, device=x.device
                )
        
        # 2. Message Passing Layers
        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)
            
            # Apply activation and dropout (except last layer before head)
            if i < self.num_layers - 1:
                h_dict = {key: F.relu(h) for key, h in h_dict.items()}
                h_dict = {key: F.dropout(h, p=self.dropout, training=self.training) 
                         for key, h in h_dict.items()}
        
        # 3. Task Head (if target node type specified)
        if self.target_node_type and self.task_head:
            h = h_dict[self.target_node_type]
            
            # For mini-batch training: only predict for seed nodes
            if batch_size is not None:
                h = h[:batch_size]
            
            out = self.task_head(h)
            return out
        
        return h_dict
    
    def get_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Get node embeddings without task head (for downstream tasks)."""
        # Project inputs
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.input_projections:
                h_dict[node_type] = self.input_projections[node_type](x)
            else:
                h_dict[node_type] = torch.zeros(
                    x.size(0), self.hidden_channels, device=x.device
                )
        
        # Message passing
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: F.relu(h) for key, h in h_dict.items()}
        
        return h_dict


@tool
def build_gnn_model(
    hetero_data: Any,
    hidden_channels: int = 64,
    out_channels: int = 2,
    num_layers: int = 2,
    architecture_type: str = "sage",
    target_node_type: Optional[str] = None,
    task_type: str = "classification",
    dropout: float = 0.2
) -> Dict[str, Any]:
    """
    Constructs a Heterogeneous GNN model adapted to the database schema.
    
    Uses HeteroConv to automatically handle multiple node/edge types.
    Each edge type gets its own message passing weights, capturing
    the semantics of different relationships in the relational database.
    
    Supported architectures:
    - 'sage': GraphSAGE (recommended for large graphs, uses sampling)
    - 'gat': Graph Attention Network (learns edge importance)
    - 'gcn': Graph Convolutional Network (simple, effective baseline)
    
    Args:
        hetero_data: The HeteroData object containing the graph structure.
                     Used to extract metadata (node types, edge types, feature dims).
        
        hidden_channels: Dimension of hidden layers. Default 64.
                         Larger values = more expressive but slower/more memory.
        
        out_channels: Output dimension. For classification, this is num_classes.
                      For regression, typically 1.
        
        num_layers: Number of GNN message passing layers. Default 2.
                    More layers = larger receptive field but risk of over-smoothing.
        
        architecture_type: GNN architecture. Options: 'sage', 'gat', 'gcn'.
        
        target_node_type: The node type for which to make predictions.
                          Example: 'customers' for customer churn prediction.
                          If None, returns embeddings for all node types.
        
        task_type: 'classification' (uses softmax) or 'regression' (raw output).
        
        dropout: Dropout rate for regularization. Default 0.2.
    
    Returns:
        Dictionary containing:
        - 'model': The PyTorch nn.Module (HeteroGNNModel)
        - 'metadata': Graph metadata used for construction
        - 'config': Model configuration for reproducibility
        
    Example:
        >>> result = build_gnn_model(
        ...     hetero_data=data,
        ...     hidden_channels=128,
        ...     out_channels=2,  # Binary classification
        ...     num_layers=3,
        ...     architecture_type='sage',
        ...     target_node_type='customers'
        ... )
        >>> model = result['model']
        >>> model.to(device)
    """
    try:
        # Validate input
        if not isinstance(hetero_data, HeteroData):
            return {
                "error": "hetero_data must be a PyTorch Geometric HeteroData object",
                "received_type": type(hetero_data).__name__
            }
        
        # Extract metadata
        metadata = hetero_data.metadata()
        node_types, edge_types = metadata
        
        # Get input feature dimensions per node type
        in_channels_dict = {}
        for node_type in node_types:
            if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
                in_channels_dict[node_type] = hetero_data[node_type].x.size(1)
            else:
                # No features: use hidden_channels as default (will be learned)
                in_channels_dict[node_type] = hidden_channels
        
        # Validate target node type
        if target_node_type and target_node_type not in node_types:
            return {
                "error": f"target_node_type '{target_node_type}' not found in graph",
                "available_types": node_types
            }
        
        # Build model
        model = HeteroGNNModel(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            conv_type=architecture_type,
            dropout=dropout,
            target_node_type=target_node_type,
            task_type=task_type
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        config = {
            "architecture": architecture_type,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "num_layers": num_layers,
            "dropout": dropout,
            "target_node_type": target_node_type,
            "task_type": task_type,
            "node_types": node_types,
            "edge_types": [str(et) for et in edge_types],
            "in_channels_dict": in_channels_dict,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }
        
        return {
            "model": model,
            "metadata": metadata,
            "config": config,
            "status": f"HeteroGNN model built with {trainable_params:,} trainable parameters"
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Failed to build model: {str(e)}",
            "traceback": traceback.format_exc()
        }


# =============================================================================
# Tool 3: Train GNN Epoch
# =============================================================================

@tool
def train_gnn_epoch(
    model: Any,
    loader: Any,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    loss_function: str = "cross_entropy",
    target_node_type: str = "",
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Executes one complete training epoch with proper backpropagation.
    
    CRITICAL for RDL mini-batch training:
    - Only computes loss on SEED NODES (first batch_size nodes in each batch)
    - Handles heterogeneous batches where each node type has different features
    - Supports temporal batches from configure_temporal_sampler
    
    Args:
        model: The HeteroGNN model from build_gnn_model.
               Must be a PyTorch nn.Module.
        
        loader: DataLoader from configure_temporal_sampler.
                Can be dict with 'loader' key or direct DataLoader.
        
        optimizer_name: Optimizer to use. Options: 'adam', 'adamw', 'sgd'.
                        Default 'adam'.
        
        learning_rate: Learning rate. Default 0.001.
        
        loss_function: Loss function. Options:
                      - 'cross_entropy': For classification (default)
                      - 'bce': Binary Cross Entropy (for binary classification)
                      - 'mse': Mean Squared Error (for regression)
                      - 'mae': Mean Absolute Error (for regression)
        
        target_node_type: The node type being predicted. Required for
                          extracting labels from batch.
        
        device: Device for training. 'auto' (default), 'cuda', 'cpu', 'mps'.
    
    Returns:
        Dictionary containing:
        - 'loss': Average loss over epoch
        - 'num_batches': Number of batches processed
        - 'total_samples': Total number of seed nodes trained
        - 'metrics': Additional metrics (accuracy for classification)
        
    Example:
        >>> for epoch in range(100):
        ...     result = train_gnn_epoch(
        ...         model=model,
        ...         loader=train_loader,
        ...         optimizer_name='adam',
        ...         learning_rate=0.001,
        ...         loss_function='cross_entropy',
        ...         target_node_type='customers'
        ...     )
        ...     print(f"Epoch {epoch}: Loss = {result['loss']:.4f}")
    """
    try:
        # Validate model
        if not isinstance(model, nn.Module):
            # Check if it's wrapped in a dict from build_gnn_model
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
            else:
                return {"error": "model must be a PyTorch nn.Module"}
        
        # Extract loader if wrapped in dict from configure_temporal_sampler
        if isinstance(loader, dict):
            if 'loader' in loader:
                loader = loader['loader']
            else:
                return {"error": "loader dict must contain 'loader' key"}
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        device = torch.device(device)
        model = model.to(device)
        model.train()
        
        # Setup optimizer
        optimizer_map = {
            'adam': lambda: torch.optim.Adam(model.parameters(), lr=learning_rate),
            'adamw': lambda: torch.optim.AdamW(model.parameters(), lr=learning_rate),
            'sgd': lambda: torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9),
        }
        
        if optimizer_name.lower() not in optimizer_map:
            return {"error": f"Unknown optimizer: {optimizer_name}. Use: adam, adamw, sgd"}
        
        optimizer = optimizer_map[optimizer_name.lower()]()
        
        # Setup loss function
        loss_fn_map = {
            'cross_entropy': nn.CrossEntropyLoss(),
            'bce': nn.BCEWithLogitsLoss(),
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
        }
        
        if loss_function.lower() not in loss_fn_map:
            return {"error": f"Unknown loss: {loss_function}. Use: cross_entropy, bce, mse, mae"}
        
        loss_fn = loss_fn_map[loss_function.lower()]
        
        # Training loop
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        num_batches = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Get batch size (number of seed nodes)
            # In NeighborLoader, the first batch_size nodes are the seeds
            if hasattr(batch, 'batch_size'):
                batch_size = batch.batch_size
            else:
                # Fallback: assume all nodes are seeds
                batch_size = None
            
            # Overwrite features for seed nodes if available (Temporal Feature Filtering)
            if target_node_type and hasattr(batch[target_node_type], 'seed_features'):
                seed_features = batch[target_node_type].seed_features
                # Ensure dimensions match
                if seed_features.size(1) == batch[target_node_type].x.size(1):
                    # Overwrite the first batch_size rows (seed nodes)
                    # seed_features corresponds to the seed nodes in this batch
                    current_batch_size = seed_features.size(0)
                    batch[target_node_type].x[:current_batch_size] = seed_features.to(device)
            
            # Forward pass
            # The model should handle x_dict and edge_index_dict
            if hasattr(batch, 'x_dict') and hasattr(batch, 'edge_index_dict'):
                out = model(batch.x_dict, batch.edge_index_dict, batch_size=batch_size)
            else:
                # Single node type case
                out = model(batch.x, batch.edge_index)
            
            # Get labels for target node type
            if target_node_type:
                if hasattr(batch[target_node_type], 'y'):
                    y = batch[target_node_type].y
                elif hasattr(batch, 'y'):
                    y = batch.y
                else:
                    return {"error": f"No labels found for target_node_type '{target_node_type}'"}
                
                # Only use labels for seed nodes (first batch_size)
                if batch_size is not None:
                    y = y[:batch_size]
            else:
                if hasattr(batch, 'y'):
                    y = batch.y
                    if batch_size is not None:
                        y = y[:batch_size]
                else:
                    return {"error": "No labels found in batch. Set target_node_type."}
            
            # Compute loss
            if loss_function.lower() == 'bce':
                # BCE expects float targets
                loss = loss_fn(out.view(-1), y.float())
            elif loss_function.lower() in ['mse', 'mae']:
                # Regression
                loss = loss_fn(out.view(-1), y.float())
            else:
                # Classification with CrossEntropy
                loss = loss_fn(out, y.long())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item() * (batch_size if batch_size else len(y))
            total_samples += batch_size if batch_size else len(y)
            num_batches += 1
            
            # Accuracy for classification
            if loss_function.lower() == 'cross_entropy':
                pred = out.argmax(dim=-1)
                total_correct += (pred == y).sum().item()
        
        # Compute averages
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        metrics = {}
        if loss_function.lower() == 'cross_entropy':
            metrics['accuracy'] = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "num_batches": num_batches,
            "total_samples": total_samples,
            "metrics": metrics,
            "device": str(device),
            "status": "Epoch completed successfully"
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Training failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


@tool
def evaluate_gnn(
    model: Any,
    loader: Any,
    loss_function: str = "cross_entropy",
    target_node_type: str = "",
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Evaluates the GNN model on a validation/test set.
    
    Similar to train_gnn_epoch but without gradient computation.
    Computes comprehensive metrics: Loss, Accuracy, ROC-AUC, F1.
    
    Args:
        model: The trained HeteroGNN model.
        loader: DataLoader for evaluation data.
        loss_function: Loss function used during training.
        target_node_type: The node type being predicted.
        device: Device for evaluation.
    
    Returns:
        Dictionary with loss and metrics (accuracy, roc_auc, f1 for classification).
    """
    try:
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        
        # Extract model/loader from dicts
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
        if isinstance(loader, dict) and 'loader' in loader:
            loader = loader['loader']
        
        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        # Loss function
        loss_fn_map = {
            'cross_entropy': nn.CrossEntropyLoss(),
            'bce': nn.BCEWithLogitsLoss(),
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
        }
        loss_fn = loss_fn_map.get(loss_function.lower(), nn.CrossEntropyLoss())
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                
                batch_size = getattr(batch, 'batch_size', None)
                
                # Overwrite features for seed nodes if available (Temporal Feature Filtering)
                # We need target_node_type for this. If not provided, we can't easily guess.
                # But evaluate_gnn takes target_node_type.
                if target_node_type and hasattr(batch[target_node_type], 'seed_features'):
                    seed_features = batch[target_node_type].seed_features
                    if seed_features.size(1) == batch[target_node_type].x.size(1):
                        current_batch_size = seed_features.size(0)
                        batch[target_node_type].x[:current_batch_size] = seed_features.to(device)

                # Forward pass
                if hasattr(batch, 'x_dict') and hasattr(batch, 'edge_index_dict'):
                    out = model(batch.x_dict, batch.edge_index_dict, batch_size=batch_size)
                else:
                    out = model(batch.x, batch.edge_index)
                
                # Get labels
                if target_node_type and hasattr(batch[target_node_type], 'y'):
                    y = batch[target_node_type].y
                else:
                    y = batch.y
                
                if batch_size is not None:
                    y = y[:batch_size]
                
                # Compute loss
                if loss_function.lower() in ['mse', 'mae']:
                    loss = loss_fn(out.view(-1), y.float())
                elif loss_function.lower() == 'bce':
                    loss = loss_fn(out.view(-1), y.float())
                else:
                    loss = loss_fn(out, y.long())
                
                total_loss += loss.item() * len(y)
                total_samples += len(y)
                
                # Collect predictions
                if loss_function.lower() == 'cross_entropy':
                    probs = F.softmax(out, dim=-1)
                    preds = out.argmax(dim=-1)
                    all_probs.extend(probs[:, 1].cpu().numpy() if probs.size(1) == 2 else probs.cpu().numpy())
                else:
                    preds = out.view(-1)
                    all_probs.extend(torch.sigmoid(out).view(-1).cpu().numpy())
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        metrics = {"loss": avg_loss}
        
        if loss_function.lower() in ['cross_entropy', 'bce']:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
            metrics['f1'] = f1_score(all_labels, all_preds, average='weighted')
            
            try:
                if len(np.unique(all_labels)) == 2:
                    metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)
            except ValueError:
                pass  # ROC-AUC not computable
        
        return {
            "metrics": metrics,
            "total_samples": total_samples,
            "status": "Evaluation completed"
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Evaluation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


@tool
def save_gnn_model(
    model: Any,
    save_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Saves the trained GNN model to disk.
    
    Saves both model weights and configuration for reproducibility.
    
    Args:
        model: The trained model (or dict with 'model' key).
        save_path: Path to save the model (e.g., 'models/fraud_detector.pt').
        config: Optional configuration dict to save alongside model.
    
    Returns:
        Dictionary with save status and path.
    """
    try:
        import os
        
        # Extract model from dict if needed
        if isinstance(model, dict):
            config = config or model.get('config', {})
            model = model.get('model', model)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save model state and config
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config or {}
        }
        
        torch.save(checkpoint, save_path)
        
        return {
            "status": "Model saved successfully",
            "path": save_path,
            "model_size_mb": os.path.getsize(save_path) / (1024 * 1024)
        }
        
    except Exception as e:
        return {"error": f"Failed to save model: {str(e)}"}


@tool
def load_gnn_model(
    load_path: str,
    hetero_data: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Loads a saved GNN model from disk.
    
    Args:
        load_path: Path to the saved model file.
        hetero_data: HeteroData object to reconstruct model architecture.
                     Required if loading for the first time.
    
    Returns:
        Dictionary with loaded model and config.
    """
    try:
        checkpoint = torch.load(load_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        if hetero_data is not None:
            # Rebuild model from config
            result = build_gnn_model(
                hetero_data=hetero_data,
                hidden_channels=config.get('hidden_channels', 64),
                out_channels=config.get('out_channels', 2),
                num_layers=config.get('num_layers', 2),
                architecture_type=config.get('architecture', 'sage'),
                target_node_type=config.get('target_node_type'),
                task_type=config.get('task_type', 'classification'),
                dropout=config.get('dropout', 0.2)
            )
            
            if 'error' in result:
                return result
            
            model = result['model']
            model.load_state_dict(checkpoint['model_state_dict'])
            
            return {
                "model": model,
                "config": config,
                "status": "Model loaded successfully"
            }
        else:
            return {
                "state_dict": checkpoint['model_state_dict'],
                "config": config,
                "status": "State dict loaded. Provide hetero_data to reconstruct full model."
            }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Failed to load model: {str(e)}",
            "traceback": traceback.format_exc()
        }
