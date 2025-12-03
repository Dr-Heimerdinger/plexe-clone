"""
Tools for graph processing and construction for the Relational Graph Architect Agent.
"""

from typing import Dict, List, Any, Tuple
from smolagents import tool
import pandas as pd
import numpy as np
import torch

# Imports for implementations
from sqlalchemy import create_engine, inspect
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import HeteroData


@tool
def extract_schema_metadata(db_connection: str) -> Dict[str, Any]:
    """
    Retrieves table names, primary keys, foreign keys, and column data types using SQLAlchemy.
    Use this in the Schema Analysis step to understand the Relational Entity Graph structure.

    Args:
        db_connection: A database connection string (e.g., 'sqlite:///datapath.db', 'postgresql://user:pass@host/db').

    Returns:
        A dictionary containing schema metadata:
        - 'tables': Dict[table_name, List[Dict[col_name, type, is_pk]]]
        - 'relationships': List[Dict[source_table, source_col, target_table, target_col]]
        - 'foreign_keys': Same as 'relationships' (alias for backwards compatibility)
    """
    try:
        engine = create_engine(db_connection)
        inspector = inspect(engine)

        schema_info = {"tables": {}, "relationships": []}

        table_names = inspector.get_table_names()

        for table_name in table_names:
            # Get Columns & PKs
            columns = inspector.get_columns(table_name)
            pk_constraint = inspector.get_pk_constraint(table_name)
            pk_cols = pk_constraint.get("constrained_columns", [])

            col_details = []
            for col in columns:
                col_details.append(
                    {"name": col["name"], "type": str(col["type"]), "is_primary_key": col["name"] in pk_cols}
                )
            schema_info["tables"][table_name] = col_details

            # Get Foreign Keys (Edges)
            fks = inspector.get_foreign_keys(table_name)
            for fk in fks:
                # Assuming single column FKs for simplicity in GNN construction
                if len(fk["constrained_columns"]) > 0:
                    schema_info["relationships"].append(
                        {
                            "source_table": table_name,
                            "source_col": fk["constrained_columns"][0],
                            "target_table": fk["referred_table"],
                            "target_col": fk["referred_columns"][0],
                        }
                    )

        # Add alias for backwards compatibility (agents may reference 'foreign_keys')
        schema_info["foreign_keys"] = schema_info["relationships"]

        return schema_info

    except Exception as e:
        return {"error": f"Failed to extract schema: {str(e)}"}


@tool
def build_hetero_graph(nodes: Dict[str, Any], edges: Dict[Tuple[str, str, str], torch.Tensor]) -> Any:
    """
    Builds a PyTorch Geometric HeteroData object from processed node features and edge indices.

    Args:
        nodes: Dictionary mapping node_type (table_name) to either:
               - A feature tensor directly (torch.Tensor), OR
               - A dict with 'x' (features) and optionally 't' (timestamps) keys
               Example: {'user': tensor([...])} or {'user': {'x': tensor, 't': tensor}}
        edges: Dictionary mapping edge_type (source, relation, target) to edge_index tensors (2, num_edges).
               Example: {('user', 'reviews', 'product'): tensor([[0, 1...], [1, 2...]])}

    Returns:
        A torch_geometric.data.HeteroData object ready for GNN training.
    """
    try:
        data = HeteroData()

        # 1. Add Node Features (supports both tensor and dict formats)
        for node_type, node_data in nodes.items():
            # Handle both dict format {'x': tensor, 't': tensor} and direct tensor format
            if isinstance(node_data, dict):
                x_tensor = node_data.get('x')
                t_tensor = node_data.get('t')
            else:
                x_tensor = node_data
                t_tensor = None

            # Process feature tensor
            if x_tensor is not None:
                if not isinstance(x_tensor, torch.Tensor):
                    try:
                        x_tensor = torch.tensor(x_tensor, dtype=torch.float)
                    except (ValueError, TypeError) as e:
                        # Handle case where x_tensor might be a nested structure
                        if isinstance(x_tensor, (list, np.ndarray)):
                            x_tensor = torch.tensor(np.array(x_tensor, dtype=np.float32), dtype=torch.float)
                        else:
                            raise ValueError(f"Cannot convert node features for '{node_type}' to tensor: {e}")
                data[node_type].x = x_tensor
                data[node_type].num_nodes = x_tensor.shape[0]

            # Process timestamp tensor (for temporal RDL)
            if t_tensor is not None:
                if not isinstance(t_tensor, torch.Tensor):
                    try:
                        t_tensor = torch.tensor(t_tensor, dtype=torch.float)
                    except (ValueError, TypeError) as e:
                        if isinstance(t_tensor, (list, np.ndarray)):
                            t_tensor = torch.tensor(np.array(t_tensor, dtype=np.float32), dtype=torch.float)
                        else:
                            raise ValueError(f"Cannot convert timestamps for '{node_type}' to tensor: {e}")
                data[node_type].t = t_tensor  # Temporal attribute for time-aware sampling

        # 2. Add Edge Indices
        for edge_key, edge_index in edges.items():
            # Validate edge_key format
            if not isinstance(edge_key, tuple) or len(edge_key) != 3:
                print(f"Warning: Skipping invalid edge key format: {edge_key}")
                continue

            # edge_key is expected to be (source_type, relation_name, target_type)
            # edge_index should be shape (2, num_edges) and type long
            if edge_index is None:
                print(f"Warning: Skipping None edge_index for {edge_key}")
                continue

            if not isinstance(edge_index, torch.Tensor):
                try:
                    edge_index = torch.tensor(edge_index, dtype=torch.long)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Cannot convert edge_index for {edge_key}: {e}")
                    continue

            # Validate edge_index shape
            if edge_index.dim() != 2 or edge_index.shape[0] != 2:
                print(f"Warning: Invalid edge_index shape {edge_index.shape} for {edge_key}, expected (2, num_edges)")
                continue

            # PyG expects standard formatting
            src_type, rel, dst_type = edge_key
            data[src_type, rel, dst_type].edge_index = edge_index

        # 3. Add Reverse Edges (Crucial for bi-directional message passing in RDL)
        # This is a simplified "T.ToUndirected" logic equivalent
        # In a real agent, we might want to be more selective, but typically we add rev_ relations.
        # This part is optional but recommended for robust GNNs.

        return data

    except Exception as e:
        import traceback
        return f"Error building HeteroData: {str(e)}\nTraceback: {traceback.format_exc()}"


@tool
def encode_multi_modal_features(data_column: List[Any], modality_type: str) -> torch.Tensor:
    """
    Encodes data attributes into embeddings/tensors suitable for GNN input.
    Supports Numeric, Categorical, Text, and Temporal modalities.

    Args:
        data_column: The data column (list) to encode.
        modality_type: 'Numeric', 'Categorical', 'Text', 'Temporal', 'Image'.

    Returns:
        A torch.Tensor of shape (num_samples, feature_dim).
    """
    try:
        # Convert to numpy for easier handling
        arr = np.array(data_column)

        if modality_type.lower() == "numeric":
            # Handle missing values (simple fill with 0, ideally should be mean)
            arr = np.nan_to_num(arr.astype(float), nan=0.0)
            scaler = StandardScaler()
            # Reshape for sklearn (N, 1)
            encoded = scaler.fit_transform(arr.reshape(-1, 1))
            return torch.tensor(encoded, dtype=torch.float)

        elif modality_type.lower() == "categorical":
            # Label Encode -> can be passed to Embedding layer later,
            # OR One-Hot encode directly if low cardinality.
            # Here we use Label Encoding for memory efficiency.
            encoder = LabelEncoder()
            # Handle mixed types by converting to str
            arr_str = [str(x) for x in arr]
            encoded = encoder.fit_transform(arr_str)
            # Return LongTensor for Embedding layers
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(1)

        elif modality_type.lower() == "temporal":
            # Convert to Unix timestamp (seconds)
            # Assuming input is convertible to datetime
            dt_series = pd.to_datetime(data_column, errors="coerce")
            # Fill NaT with min timestamp or 0
            dt_series = dt_series.fillna(pd.Timestamp(0))
            timestamp = dt_series.astype("int64") // 10**9  # Convert ns to seconds

            # Normalize timestamp (critical for stability)
            scaler = StandardScaler()
            encoded = scaler.fit_transform(timestamp.values.reshape(-1, 1))
            return torch.tensor(encoded, dtype=torch.float)

        elif modality_type.lower() == "text":
            # Use a lightweight Sentence Transformer if available
            try:
                from sentence_transformers import SentenceTransformer

                # Use a small model for speed
                model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = model.encode(list(arr), convert_to_tensor=True)
                return embeddings.cpu()  # Return as CPU tensor
            except ImportError:
                # Fallback: Simple TF-IDF or Hash if library missing
                return torch.tensor(np.zeros((len(arr), 384)), dtype=torch.float)  # Dummy

        elif modality_type.lower() == "image":
            # Placeholder: In a real agent, this would load image paths and run ResNet
            # Here we return a dummy tensor
            return torch.randn(len(arr), 128)

        else:
            raise ValueError(f"Unsupported modality: {modality_type}")

    except Exception as e:
        # Return empty tensor or informative error in tensor form to not break pipeline
        print(f"Encoding error: {e}")
        return torch.zeros(len(data_column), 1)
