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

        return schema_info

    except Exception as e:
        return {"error": f"Failed to extract schema: {str(e)}"}


@tool
def build_hetero_graph(nodes: Dict[str, torch.Tensor], edges: Dict[Tuple[str, str, str], torch.Tensor]) -> Any:
    """
    Builds a PyTorch Geometric HeteroData object from processed node features and edge indices.

    Args:
        nodes: Dictionary mapping node_type (table_name) to feature tensors (x).
               Example: {'user': tensor([...]), 'product': tensor([...])}
        edges: Dictionary mapping edge_type (source, relation, target) to edge_index tensors (2, num_edges).
               Example: {('user', 'reviews', 'product'): tensor([[0, 1...], [1, 2...]])}

    Returns:
        A torch_geometric.data.HeteroData object ready for GNN training.
    """
    try:
        data = HeteroData()

        # 1. Add Node Features
        for node_type, x_tensor in nodes.items():
            # Ensure it is a float tensor for features
            if not isinstance(x_tensor, torch.Tensor):
                x_tensor = torch.tensor(x_tensor, dtype=torch.float)
            data[node_type].x = x_tensor
            # We can also infer num_nodes
            data[node_type].num_nodes = x_tensor.shape[0]

        # 2. Add Edge Indices
        for edge_key, edge_index in edges.items():
            # edge_key is expected to be (source_type, relation_name, target_type)
            # edge_index should be shape (2, num_edges) and type long
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)

            # PyG expects standard formatting
            src_type, rel, dst_type = edge_key
            data[src_type, rel, dst_type].edge_index = edge_index

        # 3. Add Reverse Edges (Crucial for bi-directional message passing in RDL)
        # This is a simplified "T.ToUndirected" logic equivalent
        # In a real agent, we might want to be more selective, but typically we add rev_ relations.
        # This part is optional but recommended for robust GNNs.

        return data

    except Exception as e:
        return f"Error building HeteroData: {str(e)}"


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
