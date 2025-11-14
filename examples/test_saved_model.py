"""
Script to test a saved Plexe model by loading it and running predictions.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plexe

# Load the saved model
print("Loading saved model from house-prices.tar.gz...")
model = plexe.load_model("house-prices.tar.gz")
print("\nModel loaded successfully!")

# Print and save transformer code if available
if model.feature_transformer_source:
    print("\nFeature transformer code found:")
    print("-" * 80)
    print(model.feature_transformer_source)
    print("-" * 80)

    # Save transformer code for inspection
    with open("feature_transformer_code.py", "w") as f:
        f.write(model.feature_transformer_source)
    print("\nSaved transformer code to 'feature_transformer_code.py'")

# Load and prepare test data
print("\nLoading test data...")
test_df = pd.read_csv("examples/datasets/house-prices-test.csv").sample(10)
original_test = test_df.copy()  # Keep original for comparison
print(f"Test data shape: {test_df.shape}")

# Basic preprocessing for categorical columns
print("\nPreprocessing categorical columns...")
categorical_columns = test_df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    # First try ordinal encoding
    le = LabelEncoder()
    try:
        test_df[col] = le.fit_transform(test_df[col].astype(str))
    except Exception:
        # If fails, try one-hot encoding
        print(f"Using one-hot encoding for {col}")
        dummies = pd.get_dummies(test_df[col], prefix=col)
        test_df = pd.concat([test_df, dummies], axis=1)
        test_df.drop(col, axis=1, inplace=True)

# Fill missing values
print("Handling missing values...")
numeric_columns = test_df.select_dtypes(include=[np.number]).columns
test_df[numeric_columns] = test_df[numeric_columns].fillna(test_df[numeric_columns].mean())

# Try to transform test data if possible
print("\nPreparing test data for predictions...")
try:
    if hasattr(model.predictor, "transform"):
        print("Found transform method, applying model's transformations...")
        transformed_test = model.predictor.transform(original_test)  # Try with original data first
        print("Data transformed successfully by model's transformer!")
        predictions = pd.DataFrame.from_records([model.predict(x) for x in transformed_test.to_dict(orient="records")])
    else:
        print("Using manually preprocessed data...")
        predictions = pd.DataFrame.from_records([model.predict(x) for x in test_df.to_dict(orient="records")])
except Exception as e:
    print(f"\nError with model's transformer: {str(e)}")
    print("Attempting prediction with manually preprocessed data...")
    try:
        predictions = pd.DataFrame.from_records([model.predict(x) for x in test_df.to_dict(orient="records")])
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

# Print results
print("\nPredictions:")
print(predictions)

print("\nModel description:")
description = model.describe()
print(description.as_text())

# Print feature importance if available in model metadata
if hasattr(model, "metadata") and "feature_importance" in model.metadata:
    print("\nFeature Importance:")
    print(model.metadata["feature_importance"])

# Print available components
print("\nModel components available:")
print(f"Training code available: {bool(model.trainer_source)}")
print(f"Feature transformer code available: {bool(model.feature_transformer_source)}")
print(f"Testing code available: {bool(model.testing_source)}")
print(f"Dataset splitter code available: {bool(model.dataset_splitter_source)}")

# If we have the original test data with SalePrice, show comparison
if "SalePrice" in original_test.columns:
    print("\nSample comparison (actual vs predicted):")
    comparison = pd.DataFrame(
        {
            "Actual": original_test["SalePrice"],
            "Predicted": predictions["SalePrice"] if "SalePrice" in predictions.columns else predictions.iloc[:, 0],
        }
    )
    print(comparison)

    # Calculate error metrics
    errors = comparison["Actual"] - comparison["Predicted"]
    metrics = {
        "Mean Absolute Error": abs(errors).mean(),
        "Mean Squared Error": (errors**2).mean(),
        "Root Mean Squared Error": (errors**2).mean() ** 0.5,
        "Mean Absolute Percentage Error": (abs(errors) / comparison["Actual"]).mean() * 100,
    }

    print("\nError Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}")
