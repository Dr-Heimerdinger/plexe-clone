"""
Helper functions for handling features in model prediction.
"""

from typing import Dict, List, Any
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_required_features(training_data: pd.DataFrame) -> List[str]:
    """Get list of all features required for prediction based on training data."""
    categorical_cols = training_data.select_dtypes(include=["object", "category"]).columns
    numerical_cols = training_data.select_dtypes(include=["int64", "float64"]).columns

    # Get all one-hot encoded column names for categorical features
    onehot_cols = []
    for col in categorical_cols:
        unique_values = training_data[col].unique()
        onehot_cols.extend([f"{col}_{val}" for val in unique_values])

    return list(numerical_cols) + onehot_cols


def ensure_features_present(input_data: Dict[str, Any], required_features: List[str]) -> Dict[str, Any]:
    """
    Ensure all required features are present in input data.
    Args:
        input_data: Input dictionary for prediction
        required_features: List of features that should be present
    Returns:
        Dict with all required features (missing ones set to 0)
    """
    processed = input_data.copy()

    # Add missing features with default value 0
    for feature in required_features:
        if feature not in processed:
            processed[feature] = 0
            logger.debug(f"Added missing feature '{feature}' with default value 0")

    return processed


def validate_required_features(input_data: Dict[str, Any], required_features: List[str]) -> None:
    """
    Validate that input data has all strictly required features (non-one-hot).
    Raises ValueError if any required feature is missing.
    """
    # Identify base features (before one-hot encoding)
    base_features = {f.split("_")[0] for f in required_features}

    # Check for missing base features
    missing = [f for f in base_features if not any(k.startswith(f) for k in input_data.keys())]
    if missing:
        raise ValueError(f"Missing required base features: {missing}")
