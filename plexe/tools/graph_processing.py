"""
Tools for graph processing and construction for the Relational Graph Architect Agent.
"""

import re
from typing import Dict, List, Any, Tuple, Optional, Union
from smolagents import tool
import pandas as pd
import numpy as np
import torch

# Imports for implementations
from sqlalchemy import create_engine, inspect, text
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import HeteroData


# =============================================================================
# Type Definitions for Agent Clarity
# =============================================================================
# These type aliases help LLM Agents understand expected data structures.
# When passing data via REST API or non-Python environments, use List/Dict.

# Node features can be provided as Python List, NumPy array, or PyTorch Tensor
NodeFeatures = Union[List[List[float]], np.ndarray, torch.Tensor]

# Node data structure: either just features, or dict with 'x' (features) and 't' (timestamps)
NodeData = Union[
    NodeFeatures,  # Direct features tensor/list
    Dict[str, NodeFeatures]  # {'x': features, 't': timestamps}
]

# Nodes dictionary maps node type (table name) to node data
NodesDict = Dict[str, NodeData]

# Edge index: shape (2, num_edges) - source indices in row 0, target indices in row 1
EdgeIndex = Union[List[List[int]], np.ndarray, torch.Tensor]

# Edge type tuple: (source_node_type, relation_name, target_node_type)
EdgeType = Tuple[str, str, str]

# Edges dictionary maps edge types to edge indices
EdgesDict = Dict[EdgeType, EdgeIndex]


def _normalize_sql_type(sql_type: str) -> Dict[str, Any]:
    """
    Normalize SQL type string to a standardized modality category.
    
    Following RDL paper Section 3.4.3 (Multi-modal Node Encoders):
    - Numeric: For continuous values (integers, floats, decimals)
    - Categorical: For discrete values with limited cardinality (enums, small varchars)
    - Temporal: For temporal data (timestamps, dates, times) - Critical for RDL
    - Text: For free-form text (large varchars, text fields)
    - Binary: For binary/blob data
    
    These modality names match encode_multi_modal_features() expected values.
    
    Returns:
        Dict with 'modality', 'original_type', and 'is_temporal' flag
    """
    sql_type_upper = sql_type.upper()
    
    # Temporal types - Critical for RDL (Section 2.1: "Temporality as First-Class Citizen")
    temporal_patterns = [
        'TIMESTAMP', 'DATETIME', 'DATE', 'TIME', 
        'TIMESTAMPTZ', 'TIMETZ', 'INTERVAL'
    ]
    for pattern in temporal_patterns:
        if pattern in sql_type_upper:
            return {
                'modality': 'Temporal',
                'original_type': sql_type,
                'is_temporal': True
            }
    
    # Numerical types
    numerical_patterns = [
        'INT', 'INTEGER', 'SMALLINT', 'BIGINT', 'TINYINT',
        'FLOAT', 'DOUBLE', 'REAL', 'DECIMAL', 'NUMERIC',
        'MONEY', 'SERIAL', 'BIGSERIAL'
    ]
    for pattern in numerical_patterns:
        if pattern in sql_type_upper:
            return {
                'modality': 'Numeric',
                'original_type': sql_type,
                'is_temporal': False
            }
    
    # Boolean
    if 'BOOL' in sql_type_upper or 'BIT' in sql_type_upper:
        return {
            'modality': 'Categorical',
            'original_type': sql_type,
            'is_temporal': False
        }
    
    # Text types - check length to distinguish Categorical vs Text
    if any(t in sql_type_upper for t in ['VARCHAR', 'CHAR', 'NVARCHAR', 'NCHAR']):
        # Extract length if present, e.g., VARCHAR(255)
        match = re.search(r'\((\d+)\)', sql_type)
        if match:
            length = int(match.group(1))
            # Short strings are likely categorical (enums, codes, etc.)
            if length <= 50:
                return {
                    'modality': 'Categorical',
                    'original_type': sql_type,
                    'is_temporal': False
                }
        # Longer or unspecified length -> Text
        return {
            'modality': 'Text',
            'original_type': sql_type,
            'is_temporal': False
        }
    
    # Large text types
    if any(t in sql_type_upper for t in ['TEXT', 'CLOB', 'LONGTEXT', 'MEDIUMTEXT']):
        return {
            'modality': 'Text',
            'original_type': sql_type,
            'is_temporal': False
        }
    
    # Binary types
    if any(t in sql_type_upper for t in ['BLOB', 'BINARY', 'BYTEA', 'VARBINARY', 'IMAGE']):
        return {
            'modality': 'Binary',
            'original_type': sql_type,
            'is_temporal': False
        }
    
    # JSON types
    if 'JSON' in sql_type_upper:
        return {
            'modality': 'Text',
            'original_type': sql_type,
            'is_temporal': False
        }
    
    # UUID
    if 'UUID' in sql_type_upper:
        return {
            'modality': 'Categorical',
            'original_type': sql_type,
            'is_temporal': False
        }
    
    # ENUM types
    if 'ENUM' in sql_type_upper:
        return {
            'modality': 'Categorical',
            'original_type': sql_type,
            'is_temporal': False
        }
    
    # Default fallback
    return {
        'modality': 'Categorical',
        'original_type': sql_type,
        'is_temporal': False
    }


def _classify_table_type(
    table_name: str,
    row_count: int,
    has_temporal_columns: bool,
    temporal_column_count: int,
    is_referenced_by_fk: bool,
    has_outgoing_fks: bool
) -> str:
    """
    Classify table as Fact or Dimension following RDL paper Section 2.1.
    
    Fact Tables (Event/Transaction tables):
    - Usually large (many rows)
    - Have temporal columns (timestamps)
    - Often have foreign keys pointing to dimension tables
    - Examples: transactions, orders, events, logs
    
    Dimension Tables (Entity/Reference tables):
    - Usually smaller
    - May not have temporal columns (or only created_at/updated_at)
    - Are referenced by fact tables via foreign keys
    - Examples: users, products, categories
    
    Returns:
        'Fact', 'Dimension', or 'Unknown'
    """
    # Heuristics for classification
    score_fact = 0
    score_dim = 0
    
    # Large tables are more likely to be Fact tables
    if row_count > 10000:
        score_fact += 2
    elif row_count > 1000:
        score_fact += 1
    elif row_count < 100:
        score_dim += 2
    elif row_count < 1000:
        score_dim += 1
    
    # Multiple temporal columns suggest Fact table (event timestamps)
    if temporal_column_count >= 2:
        score_fact += 2
    elif has_temporal_columns:
        score_fact += 1
    
    # Being referenced by FKs suggests Dimension table
    if is_referenced_by_fk:
        score_dim += 2
    
    # Having outgoing FKs suggests Fact table (joins to dimensions)
    if has_outgoing_fks:
        score_fact += 1
    
    # Table name heuristics
    fact_keywords = ['transaction', 'order', 'event', 'log', 'history', 'activity', 'action', 'record']
    dim_keywords = ['user', 'customer', 'product', 'category', 'type', 'status', 'config', 'setting']
    
    table_lower = table_name.lower()
    if any(kw in table_lower for kw in fact_keywords):
        score_fact += 1
    if any(kw in table_lower for kw in dim_keywords):
        score_dim += 1
    
    # Determine classification
    if score_fact > score_dim:
        return 'Fact'
    elif score_dim > score_fact:
        return 'Dimension'
    else:
        return 'Unknown'


@tool
def extract_schema_metadata(db_connection: str) -> Dict[str, Any]:
    """
    Retrieves comprehensive schema metadata for Relational Deep Learning (RDL).
    
    This tool extracts table structures, relationships, and temporal information
    following the RDL paper principles where "Temporality is a First-Class Citizen" (Section 2.1).
    
    Key features:
    - Identifies temporal columns (τ_v) for time-aware message passing
    - Classifies tables as Fact (event) or Dimension (entity) tables
    - Normalizes data types to modalities (Numerical, Categorical, Time, Text, Binary)
    - Supports composite foreign keys for complex relationships
    - Provides row counts for sampling strategy decisions

    Args:
        db_connection: A database connection string 
                      (e.g., 'sqlite:///data.db', 'postgresql://user:pass@host/db').

    Returns:
        A dictionary containing:
        - 'tables': Dict[table_name, TableInfo] with columns, row_count, table_type, temporal_columns
        - 'relationships': List of foreign key relationships (supports composite keys)
        - 'temporal_summary': Overview of temporal structure across all tables
        - 'recommended_target_tables': Suggested tables for prediction tasks
    """
    try:
        engine = create_engine(db_connection)
        inspector = inspect(engine)

        schema_info = {
            "tables": {},
            "relationships": [],
            "temporal_summary": {
                "tables_with_timestamps": [],
                "primary_temporal_columns": {},
                "overall_date_range": {"min": None, "max": None}
            },
            "recommended_target_tables": []
        }

        table_names = inspector.get_table_names()
        
        # First pass: collect which tables are referenced by FKs
        referenced_tables = set()
        tables_with_outgoing_fks = set()
        for table_name in table_names:
            fks = inspector.get_foreign_keys(table_name)
            if fks:
                tables_with_outgoing_fks.add(table_name)
            for fk in fks:
                if fk.get("referred_table"):
                    referenced_tables.add(fk["referred_table"])

        all_min_dates = []
        all_max_dates = []

        for table_name in table_names:
            # Get Columns & PKs
            columns = inspector.get_columns(table_name)
            pk_constraint = inspector.get_pk_constraint(table_name)
            pk_cols = pk_constraint.get("constrained_columns", [])

            col_details = []
            temporal_columns = []
            
            for col in columns:
                # Normalize the SQL type to modality
                type_info = _normalize_sql_type(str(col["type"]))
                
                col_info = {
                    "name": col["name"],
                    "original_type": type_info["original_type"],
                    "modality": type_info["modality"],
                    "is_primary_key": col["name"] in pk_cols,
                    "is_temporal": type_info["is_temporal"],
                    "nullable": col.get("nullable", True)
                }
                col_details.append(col_info)
                
                # Track temporal columns
                if type_info["is_temporal"]:
                    temporal_columns.append(col["name"])

            # Get row count for table classification
            row_count = 0
            try:
                with engine.connect() as conn:
                    # Use COUNT(*) for accurate count
                    result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                    row_count = result.scalar() or 0
            except Exception as e:
                # Fallback if count fails
                row_count = -1  # Unknown
            
            # Get date ranges for temporal columns
            temporal_ranges = {}
            for temp_col in temporal_columns:
                try:
                    with engine.connect() as conn:
                        query = text(f'SELECT MIN("{temp_col}"), MAX("{temp_col}") FROM "{table_name}"')
                        result = conn.execute(query).fetchone()
                        if result and result[0] and result[1]:
                            temporal_ranges[temp_col] = {
                                "min": str(result[0]),
                                "max": str(result[1])
                            }
                            all_min_dates.append(result[0])
                            all_max_dates.append(result[1])
                except Exception:
                    pass

            # Classify table type
            table_type = _classify_table_type(
                table_name=table_name,
                row_count=row_count,
                has_temporal_columns=len(temporal_columns) > 0,
                temporal_column_count=len(temporal_columns),
                is_referenced_by_fk=table_name in referenced_tables,
                has_outgoing_fks=table_name in tables_with_outgoing_fks
            )

            # Identify primary temporal column (most likely to be the event timestamp)
            primary_temporal_col = None
            if temporal_columns:
                # Prefer columns with common timestamp names
                priority_names = ['created_at', 'timestamp', 'event_time', 'date', 'time', 'datetime']
                for priority in priority_names:
                    for tc in temporal_columns:
                        if priority in tc.lower():
                            primary_temporal_col = tc
                            break
                    if primary_temporal_col:
                        break
                # Default to first temporal column
                if not primary_temporal_col:
                    primary_temporal_col = temporal_columns[0]

            schema_info["tables"][table_name] = {
                "columns": col_details,
                "primary_keys": pk_cols,
                "row_count": row_count,
                "table_type": table_type,
                "temporal_columns": temporal_columns,
                "primary_temporal_column": primary_temporal_col,
                "temporal_ranges": temporal_ranges if temporal_ranges else None,
                "column_count": len(col_details),
                "modality_summary": {
                    modality: sum(1 for c in col_details if c["modality"] == modality)
                    for modality in ["Numerical", "Categorical", "Time", "Text", "Binary"]
                    if any(c["modality"] == modality for c in col_details)
                }
            }
            
            # Track temporal summary
            if temporal_columns:
                schema_info["temporal_summary"]["tables_with_timestamps"].append(table_name)
                if primary_temporal_col:
                    schema_info["temporal_summary"]["primary_temporal_columns"][table_name] = primary_temporal_col

            # Get Foreign Keys (Edges) - Support composite keys
            fks = inspector.get_foreign_keys(table_name)
            for fk in fks:
                if len(fk["constrained_columns"]) > 0:
                    relationship = {
                        "source_table": table_name,
                        "source_columns": fk["constrained_columns"],  # List for composite keys
                        "target_table": fk["referred_table"],
                        "target_columns": fk["referred_columns"],  # List for composite keys
                        "is_composite": len(fk["constrained_columns"]) > 1,
                        # Backwards compatibility - single column reference
                        "source_col": fk["constrained_columns"][0],
                        "target_col": fk["referred_columns"][0] if fk["referred_columns"] else None
                    }
                    schema_info["relationships"].append(relationship)

        # Calculate overall temporal range
        if all_min_dates and all_max_dates:
            schema_info["temporal_summary"]["overall_date_range"] = {
                "min": str(min(all_min_dates)),
                "max": str(max(all_max_dates))
            }

        # Recommend target tables (Dimension tables with good temporal coverage)
        for table_name, table_info in schema_info["tables"].items():
            if table_info["table_type"] == "Dimension" and table_info["row_count"] > 0:
                schema_info["recommended_target_tables"].append({
                    "table": table_name,
                    "row_count": table_info["row_count"],
                    "reason": "Dimension table - likely entity for prediction"
                })
            elif table_info["table_type"] == "Fact" and table_info["temporal_columns"]:
                schema_info["recommended_target_tables"].append({
                    "table": table_name,
                    "row_count": table_info["row_count"],
                    "reason": "Fact table with temporal data - good for event prediction"
                })

        # Add alias for backwards compatibility
        schema_info["foreign_keys"] = schema_info["relationships"]

        return schema_info

    except Exception as e:
        import traceback
        return {
            "error": f"Failed to extract schema: {str(e)}",
            "traceback": traceback.format_exc(),
            "tables": {},
            "relationships": [],
            "temporal_summary": {"tables_with_timestamps": [], "primary_temporal_columns": {}}
        }


# =============================================================================
# EntityMapper: Centralized ID Mapping for Graph Construction
# =============================================================================
# Following RDL paper: Maintains bidirectional mapping between database PKs
# and contiguous graph indices. Essential for post-prediction interpretation.

class EntityMapper:
    """
    Centralized entity mapping manager for Relational Deep Learning.
    
    This class maintains bidirectional mappings between original database IDs
    (UUIDs, strings, non-contiguous integers) and contiguous graph indices
    required by PyTorch Geometric.
    
    Key Features:
    - Bidirectional mapping: ID -> Index and Index -> ID
    - Multi-table support: Manages mappings for all entity types
    - Prediction interpretation: Map predicted node indices back to original IDs
    - Serializable: Can be saved/loaded for model deployment
    
    Example:
        >>> mapper = EntityMapper()
        >>> mapper.add_entities('customers', ['C001', 'C002', 'C003'])
        >>> mapper.add_entities('products', ['P100', 'P200'])
        >>> 
        >>> # Convert FK to edge index
        >>> customer_idx = mapper.id_to_index('customers', 'C001')  # Returns 0
        >>> product_idx = mapper.id_to_index('products', 'P200')    # Returns 1
        >>> 
        >>> # After GNN prediction: node 0 is fraud
        >>> fraud_customer = mapper.index_to_id('customers', 0)  # Returns 'C001'
    """
    
    def __init__(self):
        self._mappings: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, int] = {}
    
    def add_entities(self, entity_type: str, original_ids: List[Any]) -> Dict[str, Any]:
        """
        Register entities of a specific type and create index mappings.
        
        Args:
            entity_type: The entity/table name (e.g., 'customers', 'orders')
            original_ids: List of original database IDs
            
        Returns:
            Mapping info with id_to_idx, idx_to_id, and num_nodes
        """
        if original_ids is None or len(original_ids) == 0:
            self._mappings[entity_type] = {
                "id_to_idx": {},
                "idx_to_id": {},
                "num_nodes": 0
            }
            return self._mappings[entity_type]
        
        # Convert to list if numpy array
        if isinstance(original_ids, np.ndarray):
            original_ids = original_ids.tolist()
        
        # Create ordered unique list (preserve insertion order)
        unique_ids = []
        seen = set()
        for id_val in original_ids:
            hashable_id = str(id_val) if not isinstance(id_val, (int, str, float)) else id_val
            if hashable_id not in seen:
                unique_ids.append(id_val)
                seen.add(hashable_id)
        
        # Create bidirectional mappings
        id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}
        idx_to_id = {idx: id_val for idx, id_val in enumerate(unique_ids)}
        
        self._mappings[entity_type] = {
            "id_to_idx": id_to_idx,
            "idx_to_id": idx_to_id,
            "num_nodes": len(unique_ids)
        }
        self._stats[entity_type] = len(unique_ids)
        
        return self._mappings[entity_type]
    
    def id_to_index(self, entity_type: str, original_id: Any) -> Optional[int]:
        """Convert original database ID to contiguous graph index."""
        if entity_type not in self._mappings:
            return None
        return self._mappings[entity_type]["id_to_idx"].get(original_id)
    
    def index_to_id(self, entity_type: str, index: int) -> Optional[Any]:
        """Convert graph index back to original database ID (for prediction interpretation)."""
        if entity_type not in self._mappings:
            return None
        return self._mappings[entity_type]["idx_to_id"].get(index)
    
    def batch_id_to_index(self, entity_type: str, original_ids: List[Any]) -> List[int]:
        """Convert a batch of IDs to indices, filtering out unmapped IDs."""
        if entity_type not in self._mappings:
            return []
        id_to_idx = self._mappings[entity_type]["id_to_idx"]
        return [id_to_idx[id_val] for id_val in original_ids if id_val in id_to_idx]
    
    def batch_index_to_id(self, entity_type: str, indices: List[int]) -> List[Any]:
        """Convert a batch of indices back to original IDs."""
        if entity_type not in self._mappings:
            return []
        idx_to_id = self._mappings[entity_type]["idx_to_id"]
        return [idx_to_id[idx] for idx in indices if idx in idx_to_id]
    
    def get_mapping(self, entity_type: str) -> Optional[Dict[str, Any]]:
        """Get the full mapping for an entity type."""
        return self._mappings.get(entity_type)
    
    def get_all_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Get all mappings (for serialization)."""
        return self._mappings.copy()
    
    def get_num_nodes(self, entity_type: str) -> int:
        """Get the number of nodes for an entity type."""
        if entity_type not in self._mappings:
            return 0
        return self._mappings[entity_type]["num_nodes"]
    
    def get_entity_types(self) -> List[str]:
        """Get list of all registered entity types."""
        return list(self._mappings.keys())
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the mapper state."""
        return {
            "entity_types": self.get_entity_types(),
            "node_counts": self._stats.copy(),
            "total_nodes": sum(self._stats.values())
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize mapper to dictionary (for saving)."""
        return {
            "mappings": self._mappings,
            "stats": self._stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityMapper':
        """Deserialize mapper from dictionary (for loading)."""
        mapper = cls()
        mapper._mappings = data.get("mappings", {})
        mapper._stats = data.get("stats", {})
        return mapper


# Global mapper instance (can be used across tool calls)
_global_entity_mapper: Optional[EntityMapper] = None


def get_global_entity_mapper() -> EntityMapper:
    """Get or create the global EntityMapper instance."""
    global _global_entity_mapper
    if _global_entity_mapper is None:
        _global_entity_mapper = EntityMapper()
    return _global_entity_mapper


def reset_global_entity_mapper() -> None:
    """Reset the global EntityMapper (for new graph construction)."""
    global _global_entity_mapper
    _global_entity_mapper = EntityMapper()


# =============================================================================
# Semantic Relation Naming
# =============================================================================
# Following RDL paper: Relations should have semantic meaning.
# E.g., ('customer', 'places', 'order') not ('customer', 'fk_order_id', 'order')

def generate_semantic_relation_name(
    source_table: str,
    target_table: str,
    fk_column: str,
    is_reverse: bool = False
) -> str:
    """
    Generate a semantic relation name based on table names and FK column.
    
    Following RDL paper best practices for meaningful edge types.
    
    Args:
        source_table: Source entity type (e.g., 'orders')
        target_table: Target entity type (e.g., 'customers')
        fk_column: Foreign key column name (e.g., 'customer_id')
        is_reverse: If True, generate reverse relation name
        
    Returns:
        Semantic relation name (e.g., 'placed_by' for order->customer)
        
    Examples:
        Forward:  ('orders', 'placed_by', 'customers')
        Reverse:  ('customers', 'places', 'orders')
    """
    # Common semantic patterns based on table/column names
    semantic_patterns = {
        # Customer/User related
        ('order', 'customer'): ('placed_by', 'places'),
        ('order', 'user'): ('placed_by', 'places'),
        ('transaction', 'customer'): ('made_by', 'makes'),
        ('transaction', 'user'): ('made_by', 'makes'),
        ('review', 'customer'): ('written_by', 'writes'),
        ('review', 'user'): ('written_by', 'writes'),
        ('purchase', 'customer'): ('made_by', 'makes'),
        ('purchase', 'user'): ('made_by', 'makes'),
        
        # Product related
        ('order', 'product'): ('contains', 'ordered_in'),
        ('order_item', 'product'): ('contains', 'included_in'),
        ('review', 'product'): ('reviews', 'reviewed_by'),
        ('cart', 'product'): ('contains', 'in_cart'),
        ('wishlist', 'product'): ('contains', 'wishlisted_by'),
        
        # Category/Type related
        ('product', 'category'): ('belongs_to', 'contains'),
        ('product', 'brand'): ('manufactured_by', 'manufactures'),
        ('item', 'category'): ('belongs_to', 'contains'),
        
        # Location related
        ('order', 'address'): ('shipped_to', 'receives'),
        ('customer', 'address'): ('lives_at', 'residence_of'),
        ('user', 'address'): ('lives_at', 'residence_of'),
        ('store', 'location'): ('located_at', 'hosts'),
        
        # Payment related
        ('order', 'payment'): ('paid_with', 'pays_for'),
        ('transaction', 'payment_method'): ('uses', 'used_by'),
        
        # Employee/Staff related
        ('order', 'employee'): ('handled_by', 'handles'),
        ('task', 'employee'): ('assigned_to', 'assigned'),
        ('project', 'employee'): ('worked_on_by', 'works_on'),
        
        # Generic fallbacks
        ('item', 'order'): ('part_of', 'has_item'),
        ('detail', 'master'): ('belongs_to', 'has_detail'),
    }
    
    # Normalize table names for matching
    src_lower = source_table.lower().rstrip('s')  # Remove plural 's'
    tgt_lower = target_table.lower().rstrip('s')
    
    # Try to find a matching pattern
    key = (src_lower, tgt_lower)
    if key in semantic_patterns:
        forward_rel, reverse_rel = semantic_patterns[key]
        return reverse_rel if is_reverse else forward_rel
    
    # Try reverse key
    reverse_key = (tgt_lower, src_lower)
    if reverse_key in semantic_patterns:
        forward_rel, reverse_rel = semantic_patterns[reverse_key]
        return forward_rel if is_reverse else reverse_rel
    
    # Fallback: Generate from FK column name
    # E.g., 'customer_id' -> 'has_customer' (reverse) or 'of_customer' (forward)
    fk_clean = fk_column.lower().replace('_id', '').replace('id', '').strip('_')
    if fk_clean:
        if is_reverse:
            return f"has_{fk_clean}"
        else:
            return f"of_{fk_clean}"
    
    # Ultimate fallback
    if is_reverse:
        return f"rev_{source_table}_to_{target_table}"
    else:
        return f"{source_table}_to_{target_table}"


@tool
def create_entity_mapper() -> Dict[str, Any]:
    """
    Creates a new EntityMapper for managing ID-to-Index mappings.
    
    The EntityMapper is essential for:
    1. Converting database PKs (UUIDs, strings) to contiguous graph indices
    2. Interpreting GNN predictions back to original entity IDs
    3. Managing mappings across multiple entity types
    
    Returns:
        A dictionary confirming mapper creation with usage instructions.
        
    Example workflow:
        1. mapper_info = create_entity_mapper()
        2. register_entities('customers', customer_ids)
        3. register_entities('products', product_ids)
        4. ... build graph using mapped indices ...
        5. After prediction: interpret_prediction('customers', predicted_indices)
    """
    reset_global_entity_mapper()
    return {
        "status": "created",
        "message": "EntityMapper created. Use register_entities() to add entity mappings.",
        "next_steps": [
            "Call register_entities(entity_type, id_list) for each table",
            "Use get_entity_index(entity_type, original_id) to convert IDs",
            "After GNN prediction, use interpret_prediction() to get original IDs"
        ]
    }


@tool
def register_entities(
    entity_type: str,
    original_ids: List[Any]
) -> Dict[str, Any]:
    """
    Register entities and create ID-to-Index mapping for a table.
    
    CRITICAL: Call this for EVERY table BEFORE building edges.
    
    Args:
        entity_type: The entity/table name (e.g., 'customers', 'orders', 'products')
        original_ids: List of original Primary Key values from the database.
                      Can be UUIDs, strings, or integers.
    
    Returns:
        Mapping information including:
        - 'entity_type': The registered entity type
        - 'num_nodes': Number of unique entities
        - 'sample_mapping': First few ID->Index mappings (for verification)
        - 'id_to_idx': Full mapping dictionary (for edge construction)
        
    Example:
        >>> result = register_entities('customers', ['C001', 'C002', 'C003'])
        >>> result['num_nodes']  # 3
        >>> result['id_to_idx']['C001']  # 0
    """
    mapper = get_global_entity_mapper()
    mapping = mapper.add_entities(entity_type, original_ids)
    
    # Create sample for verification
    sample_size = min(5, mapping["num_nodes"])
    sample_mapping = dict(list(mapping["id_to_idx"].items())[:sample_size])
    
    return {
        "entity_type": entity_type,
        "num_nodes": mapping["num_nodes"],
        "sample_mapping": sample_mapping,
        "id_to_idx": mapping["id_to_idx"],
        "message": f"Registered {mapping['num_nodes']} {entity_type} entities"
    }


@tool
def get_entity_index(
    entity_type: str,
    original_id: Any
) -> Dict[str, Any]:
    """
    Convert a single original database ID to its graph index.
    
    Args:
        entity_type: The entity/table name
        original_id: The original database ID value
        
    Returns:
        Dictionary with the index or error message
    """
    mapper = get_global_entity_mapper()
    index = mapper.id_to_index(entity_type, original_id)
    
    if index is None:
        return {
            "error": f"ID '{original_id}' not found in entity type '{entity_type}'",
            "registered_types": mapper.get_entity_types()
        }
    
    return {
        "entity_type": entity_type,
        "original_id": original_id,
        "graph_index": index
    }


@tool
def convert_edge_ids_to_indices(
    source_entity_type: str,
    target_entity_type: str,
    source_ids: List[Any],
    target_ids: List[Any],
    relation_name: Optional[str] = None,
    source_fk_column: str = ""
) -> Dict[str, Any]:
    """
    Convert foreign key ID pairs to graph edge indices with semantic naming.
    
    This is the primary tool for building edges. It:
    1. Converts source/target IDs to contiguous indices
    2. Generates semantic relation names (if not provided)
    3. Returns both forward and reverse edge data
    
    Args:
        source_entity_type: Source table name (e.g., 'orders')
        target_entity_type: Target table name (e.g., 'customers')
        source_ids: List of source entity IDs (from FK column values)
        target_ids: List of target entity IDs (the referenced PKs)
        relation_name: Optional custom relation name. If None, auto-generated.
        source_fk_column: FK column name for semantic naming (e.g., 'customer_id')
        
    Returns:
        Dictionary containing:
        - 'forward_edge': Tuple (edge_type, edge_index) for source->target
        - 'reverse_edge': Tuple (edge_type, edge_index) for target->source
        - 'num_edges': Number of valid edges created
        - 'dropped_edges': Number of edges with unmapped IDs (dropped)
        
    Example:
        >>> result = convert_edge_ids_to_indices(
        ...     'orders', 'customers',
        ...     order_df['customer_id'].tolist(),  # FK values
        ...     order_df['customer_id'].tolist(),  # Same for simple FK
        ...     source_fk_column='customer_id'
        ... )
        >>> result['forward_edge']
        # (('orders', 'placed_by', 'customers'), [[0,1,2], [1,0,2]])
    """
    mapper = get_global_entity_mapper()
    
    # Validate entity types are registered
    if source_entity_type not in mapper.get_entity_types():
        return {"error": f"Entity type '{source_entity_type}' not registered. Call register_entities first."}
    if target_entity_type not in mapper.get_entity_types():
        return {"error": f"Entity type '{target_entity_type}' not registered. Call register_entities first."}
    
    # Get mappings
    src_mapping = mapper.get_mapping(source_entity_type)["id_to_idx"]
    tgt_mapping = mapper.get_mapping(target_entity_type)["id_to_idx"]
    
    # Convert IDs to indices, filtering unmapped
    src_indices = []
    tgt_indices = []
    dropped = 0
    
    for src_id, tgt_id in zip(source_ids, target_ids):
        src_idx = src_mapping.get(src_id)
        tgt_idx = tgt_mapping.get(tgt_id)
        
        if src_idx is not None and tgt_idx is not None:
            src_indices.append(src_idx)
            tgt_indices.append(tgt_idx)
        else:
            dropped += 1
    
    # Generate semantic relation names
    if relation_name:
        forward_rel = relation_name
        reverse_rel = f"rev_{relation_name}"
    else:
        forward_rel = generate_semantic_relation_name(
            source_entity_type, target_entity_type, source_fk_column, is_reverse=False
        )
        reverse_rel = generate_semantic_relation_name(
            source_entity_type, target_entity_type, source_fk_column, is_reverse=True
        )
    
    # Create edge indices (shape: 2 x num_edges)
    forward_edge_index = [src_indices, tgt_indices]
    reverse_edge_index = [tgt_indices, src_indices]
    
    return {
        "forward_edge": {
            "edge_type": (source_entity_type, forward_rel, target_entity_type),
            "edge_index": forward_edge_index,
        },
        "reverse_edge": {
            "edge_type": (target_entity_type, reverse_rel, source_entity_type),
            "edge_index": reverse_edge_index,
        },
        "num_edges": len(src_indices),
        "dropped_edges": dropped,
        "relations": {
            "forward": f"({source_entity_type}) --[{forward_rel}]--> ({target_entity_type})",
            "reverse": f"({target_entity_type}) --[{reverse_rel}]--> ({source_entity_type})"
        }
    }


@tool
def interpret_prediction(
    entity_type: str,
    predicted_indices: List[int],
    prediction_scores: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Interpret GNN prediction results by mapping indices back to original IDs.
    
    After the GNN makes predictions (e.g., fraud detection, link prediction),
    use this tool to convert the predicted node indices back to original
    database IDs for business interpretation.
    
    Args:
        entity_type: The entity type that was predicted (e.g., 'customers')
        predicted_indices: List of predicted node indices from GNN
        prediction_scores: Optional list of prediction scores/probabilities
        
    Returns:
        Dictionary with original IDs and their predictions:
        - 'predictions': List of {original_id, index, score} dicts
        - 'entity_type': The entity type
        - 'count': Number of predictions
        
    Example:
        >>> # GNN predicts nodes 0, 2, 5 as fraudulent
        >>> result = interpret_prediction('customers', [0, 2, 5], [0.95, 0.87, 0.91])
        >>> result['predictions']
        # [{'original_id': 'C001', 'index': 0, 'score': 0.95}, ...]
    """
    mapper = get_global_entity_mapper()
    
    if entity_type not in mapper.get_entity_types():
        return {"error": f"Entity type '{entity_type}' not registered."}
    
    predictions = []
    for i, idx in enumerate(predicted_indices):
        original_id = mapper.index_to_id(entity_type, idx)
        pred = {
            "original_id": original_id,
            "graph_index": idx
        }
        if prediction_scores and i < len(prediction_scores):
            pred["score"] = prediction_scores[i]
        predictions.append(pred)
    
    return {
        "entity_type": entity_type,
        "count": len(predictions),
        "predictions": predictions
    }


@tool
def get_mapper_summary() -> Dict[str, Any]:
    """
    Get a summary of the current EntityMapper state.
    
    Returns:
        Summary including registered entity types, node counts, and total nodes.
    """
    mapper = get_global_entity_mapper()
    return mapper.summary()


# Keep old function for backwards compatibility (deprecated)
@tool
def create_id_mapping(
    original_ids: List[Any],
    table_name: str = ""
) -> Dict[str, Any]:
    """
    [DEPRECATED] Use register_entities() instead.
    
    Creates a safe mapping from original database IDs to contiguous indices.
    This function is kept for backwards compatibility but register_entities()
    provides better integration with the EntityMapper system.
    """
    # Use the new EntityMapper system internally
    mapper = get_global_entity_mapper()
    mapping = mapper.add_entities(table_name or "unnamed", original_ids)
    
    return {
        "id_to_idx": mapping["id_to_idx"],
        "idx_to_id": mapping["idx_to_id"],
        "num_nodes": mapping["num_nodes"],
        "table_name": table_name,
        "deprecation_notice": "Use register_entities() for better integration with EntityMapper"
    }


@tool
def build_hetero_graph(
    nodes: NodesDict,
    edges: EdgesDict,
    add_reverse_edges: bool = True
) -> Any:
    """
    Builds a PyTorch Geometric HeteroData object from processed node features and edge indices.
    
    IMPORTANT: This tool implements bi-directional message passing as required by RDL paper.
    By default, it automatically adds reverse edges for every forward edge.
    
    Type Hints for LLM Agents:
    - nodes: Dict[str, Union[List, np.ndarray, torch.Tensor, Dict]]
      - Keys are node types (table names)
      - Values can be:
        - A 2D list/array/tensor of shape (num_nodes, feature_dim)
        - A dict with 'x' (features) and optionally 't' (timestamps)
    - edges: Dict[Tuple[str, str, str], Union[List, np.ndarray, torch.Tensor]]
      - Keys are (source_type, relation_name, target_type)
      - Values are edge indices of shape (2, num_edges)
      - Row 0: source node indices, Row 1: target node indices
    
    CRITICAL: Edge indices must be contiguous integers [0, N-1].
    Use create_id_mapping() to convert database IDs before calling this tool.

    Args:
        nodes: Dictionary mapping node_type (table_name) to node data.
               Accepts List[List[float]], numpy.ndarray, or torch.Tensor.
               Can also be a dict with 'x' (features) and 't' (timestamps) keys.
               Example: {'users': [[0.1, 0.2], [0.3, 0.4]]}
               Example: {'users': {'x': features_array, 't': timestamps_array}}
               
        edges: Dictionary mapping edge_type to edge_index.
               Accepts List[List[int]], numpy.ndarray, or torch.Tensor.
               Shape must be (2, num_edges) where row 0 is source, row 1 is target.
               Example: {('users', 'purchased', 'products'): [[0, 1, 2], [1, 2, 0]]}
               
        add_reverse_edges: If True (default), automatically adds reverse edges for
                          bi-directional message passing. The reverse relation is
                          prefixed with 'rev_'. Set to False if you manually handle
                          reverse edges.

    Returns:
        A torch_geometric.data.HeteroData object ready for GNN training.
        Contains:
        - data[node_type].x: Node feature tensor
        - data[node_type].t: Timestamp tensor (if provided)
        - data[node_type].num_nodes: Number of nodes
        - data[src, rel, dst].edge_index: Edge connections
        - data[dst, 'rev_' + rel, src].edge_index: Reverse edges (if add_reverse_edges=True)
    """
    try:
        data = HeteroData()

        # 1. Add Node Features (supports both tensor and dict formats)
        for node_type, node_data in nodes.items():
            # Handle both dict format {'x': tensor, 't': tensor} and direct tensor format
            if isinstance(node_data, dict):
                x_tensor = node_data.get('x')
                t_tensor = node_data.get('t')
                
                # Support output from encode_multi_modal_features {'tensor': ..., 'metadata': ...}
                if x_tensor is None and 'tensor' in node_data:
                    x_tensor = node_data.get('tensor')
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

        # 2. Add Edge Indices and Reverse Edges
        reverse_edges_added = []
        
        for edge_key, edge_index in edges.items():
            # Validate edge_key format
            if not isinstance(edge_key, tuple) or len(edge_key) != 3:
                print(f"Warning: Skipping invalid edge key format: {edge_key}")
                continue

            # edge_key is expected to be (source_type, relation_name, target_type)
            src_type, rel, dst_type = edge_key
            
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

            # Validate edge indices are within bounds
            if data[src_type].num_nodes is not None:
                max_src_idx = edge_index[0].max().item() if edge_index.shape[1] > 0 else -1
                if max_src_idx >= data[src_type].num_nodes:
                    print(f"Warning: Source index {max_src_idx} out of bounds for {src_type} (has {data[src_type].num_nodes} nodes)")
            
            if data[dst_type].num_nodes is not None:
                max_dst_idx = edge_index[1].max().item() if edge_index.shape[1] > 0 else -1
                if max_dst_idx >= data[dst_type].num_nodes:
                    print(f"Warning: Target index {max_dst_idx} out of bounds for {dst_type} (has {data[dst_type].num_nodes} nodes)")

            # Add forward edge
            data[src_type, rel, dst_type].edge_index = edge_index

            # 3. Add Reverse Edges (Crucial for bi-directional message passing in RDL)
            # Following RDL paper: GNNs need to propagate information in both directions
            if add_reverse_edges:
                # Generate semantic reverse relation name
                # If relation already has semantic name, generate appropriate reverse
                reverse_rel = generate_semantic_relation_name(
                    src_type, dst_type, rel, is_reverse=True
                )
                # Fallback if semantic naming returns same as forward
                if reverse_rel == rel:
                    reverse_rel = f"rev_{rel}"
                
                # Flip edge_index: swap source and target
                reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
                data[dst_type, reverse_rel, src_type].edge_index = reverse_edge_index
                reverse_edges_added.append((dst_type, reverse_rel, src_type))

        return data

    except Exception as e:
        import traceback
        return f"Error building HeteroData: {str(e)}\nTraceback: {traceback.format_exc()}"


@tool
def encode_multi_modal_features(
    data_column: List[Any], 
    modality_type: str,
    encoding_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Encodes data attributes into tensors with metadata, handling missing values and type conflicts.
    
    Addresses common GNN pitfalls:
    1. Type Conflicts: Returns explicit tensor types and metadata for Embedding layers.
    2. Temporal: Uses cyclical encoding (sin/cos) to capture periodic patterns.
    3. Missing Data: Uses imputation + missing indicators instead of zero-filling.
    
    Args:
        data_column: List of data values.
        modality_type: 'Numeric', 'Categorical', 'Temporal', 'Text', 'Image'.
        encoding_options: Optional settings.
            - 'categorical_strategy': 'auto', 'one_hot', 'embedding'.
            - 'handle_missing': 'mean', 'median', 'constant'.
            
    Returns:
        Dict containing:
        - 'tensor': The encoded torch.Tensor.
        - 'tensor_type': 'float' (for direct use) or 'long' (requires embedding).
        - 'metadata': Dict with 'num_classes', 'feature_dim', etc.
    """
    try:
        if encoding_options is None:
            encoding_options = {}
            
        # Convert input to pandas Series for robust handling
        series = pd.Series(data_column)
        num_samples = len(series)
        
        # Result container
        result = {
            "tensor": None,
            "tensor_type": "float",
            "metadata": {}
        }

        if modality_type.lower() == "numeric":
            # 1. Handle Missing Values explicitly
            # Coerce to numeric, turning errors to NaN
            series_num = pd.to_numeric(series, errors='coerce')
            
            # Create missing indicator (1.0 if missing, 0.0 otherwise)
            is_missing = series_num.isna().astype(float).values.reshape(-1, 1)
            
            # Impute missing values
            strategy = encoding_options.get('handle_missing', 'mean')
            if strategy == 'mean':
                fill_val = series_num.mean()
            elif strategy == 'median':
                fill_val = series_num.median()
            else:
                fill_val = 0.0
                
            if pd.isna(fill_val): # Handle all-NaN case
                fill_val = 0.0
                
            series_filled = series_num.fillna(fill_val).values.reshape(-1, 1)
            
            # Scale values
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(series_filled)
            
            # Concatenate [Value, MissingIndicator]
            # This allows the model to distinguish "true 0" from "missing imputed as 0"
            final_tensor = torch.tensor(np.hstack([scaled_values, is_missing]), dtype=torch.float)
            
            result["tensor"] = final_tensor
            result["tensor_type"] = "float"
            result["metadata"] = {"feature_dim": 2, "description": "Scaled Value + Missing Indicator"}

        elif modality_type.lower() == "categorical":
            # Handle missing values as a specific category
            series_cat = series.fillna("MISSING").astype(str)
            unique_vals = series_cat.unique()
            num_classes = len(unique_vals)
            
            # Determine strategy
            strategy = encoding_options.get('categorical_strategy', 'auto')
            if strategy == 'auto':
                # Use One-Hot for low cardinality, Embedding for high
                strategy = 'one_hot' if num_classes <= 20 else 'embedding'
            
            if strategy == 'one_hot':
                # One-Hot Encoding -> FloatTensor
                dummies = pd.get_dummies(series_cat, prefix='cat')
                final_tensor = torch.tensor(dummies.values, dtype=torch.float)
                
                result["tensor"] = final_tensor
                result["tensor_type"] = "float"
                result["metadata"] = {
                    "feature_dim": final_tensor.shape[1],
                    "strategy": "one_hot",
                    "classes": list(dummies.columns)
                }
                
            else: # embedding
                # Label Encoding -> LongTensor
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(series_cat)
                final_tensor = torch.tensor(encoded, dtype=torch.long)
                
                result["tensor"] = final_tensor
                result["tensor_type"] = "long"
                result["metadata"] = {
                    "num_classes": num_classes,
                    "strategy": "embedding",
                    "classes": list(encoder.classes_)
                }

        elif modality_type.lower() == "temporal":
            # Convert to datetime
            dt_series = pd.to_datetime(series, errors="coerce")
            
            # Handle missing
            is_missing = dt_series.isna().astype(float).values.reshape(-1, 1)
            
            # Fill NaT with mean time (or min) to allow extraction
            mean_time = dt_series.dropna().mean()
            if pd.isna(mean_time):
                mean_time = pd.Timestamp.now()
            dt_filled = dt_series.fillna(mean_time)
            
            features_list = [is_missing]
            
            # 1. Linear Time (Scaled Timestamp)
            timestamp = dt_filled.astype("int64") // 10**9
            scaler = StandardScaler()
            scaled_time = scaler.fit_transform(timestamp.values.reshape(-1, 1))
            features_list.append(scaled_time)
            
            # 2. Cyclical Features (Sin/Cos)
            # Hour of day (0-23)
            hours = dt_filled.dt.hour.values
            features_list.append(np.sin(2 * np.pi * hours / 24).reshape(-1, 1))
            features_list.append(np.cos(2 * np.pi * hours / 24).reshape(-1, 1))
            
            # Day of week (0-6)
            dow = dt_filled.dt.dayofweek.values
            features_list.append(np.sin(2 * np.pi * dow / 7).reshape(-1, 1))
            features_list.append(np.cos(2 * np.pi * dow / 7).reshape(-1, 1))
            
            # Month (1-12)
            months = dt_filled.dt.month.values
            features_list.append(np.sin(2 * np.pi * (months-1) / 12).reshape(-1, 1))
            features_list.append(np.cos(2 * np.pi * (months-1) / 12).reshape(-1, 1))
            
            # Concatenate all
            final_array = np.hstack(features_list)
            final_tensor = torch.tensor(final_array, dtype=torch.float)
            
            result["tensor"] = final_tensor
            result["tensor_type"] = "float"
            result["metadata"] = {
                "feature_dim": final_tensor.shape[1],
                "components": ["is_missing", "scaled_time", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
            }

        elif modality_type.lower() == "text":
            # Use Sentence Transformer
            try:
                from sentence_transformers import SentenceTransformer
                # Use a small model for speed
                model = SentenceTransformer("all-MiniLM-L6-v2")
                # Handle missing text
                text_list = series.fillna("").astype(str).tolist()
                embeddings = model.encode(text_list, convert_to_tensor=True)
                
                result["tensor"] = embeddings.cpu()
                result["tensor_type"] = "float"
                result["metadata"] = {"feature_dim": embeddings.shape[1], "model": "all-MiniLM-L6-v2"}
                
            except ImportError:
                # Fallback: Simple Hash encoding if library missing
                print("Warning: sentence_transformers not found. Using dummy hash encoding.")
                # Create random but consistent embeddings based on hash
                # This is just a fallback to prevent crash
                feature_dim = 64
                text_list = series.fillna("").astype(str).tolist()
                # Simple hashing trick
                hashed = np.array([hash(s) % 10000 for s in text_list]).reshape(-1, 1)
                scaler = StandardScaler()
                encoded = scaler.fit_transform(hashed)
                # Expand to feature_dim (just repeating for dummy)
                encoded = np.tile(encoded, (1, feature_dim))
                
                result["tensor"] = torch.tensor(encoded, dtype=torch.float)
                result["tensor_type"] = "float"
                result["metadata"] = {"feature_dim": feature_dim, "note": "Fallback Hash Encoding"}

        elif modality_type.lower() == "image":
            # Placeholder
            result["tensor"] = torch.randn(num_samples, 128)
            result["tensor_type"] = "float"
            result["metadata"] = {"feature_dim": 128, "note": "Placeholder Random Features"}

        else:
            raise ValueError(f"Unsupported modality: {modality_type}")

        return result

    except Exception as e:
        import traceback
        print(f"Encoding error for {modality_type}: {e}")
        print(traceback.format_exc())
        # Return safe fallback
        return {
            "tensor": torch.zeros(len(data_column), 1),
            "tensor_type": "float",
            "metadata": {"error": str(e), "fallback": True}
        }
