import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from plexe.core.interfaces.feature_transformer import FeatureTransformer


class FeatureTransformerImplementation(FeatureTransformer):

    def transform(self, inputs: pd.DataFrame) -> pd.DataFrame:
        df = inputs.copy()

        target_variable = "SalePrice"

        # 1. Target Variable Transformation
        if target_variable in df.columns:
            df[target_variable] = np.log1p(df[target_variable])
            print(f"Applied log1p transformation to target variable '{target_variable}'.")
        else:
            print(f"Target variable '{target_variable}' not found in DataFrame. Skipping transformation.")

        # Separate features and target for transformation (if target is present)
        if target_variable in df.columns:
            X = df.drop(columns=[target_variable])
            y = df[target_variable]
        else:
            X = df
            y = None  # No target present

        # Identify numerical and categorical features based on dtypes

        # Define features for specific missing value imputation based on EDA
        numerical_features_with_nan_specific = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
        categorical_features_with_nan_specific = [
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "FireplaceQu",
            "GarageType",
            "GarageFinish",
            "GarageQual",
            "GarageCond",
        ]

        # Step 1: Impute numerical missing values using median
        for col in numerical_features_with_nan_specific:
            if col in X.columns:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"Imputed missing values in numerical feature '{col}' with median: {median_val}.")

        # Step 2: Impute categorical missing values using 'NA'
        for col in categorical_features_with_nan_specific:
            if col in X.columns:
                X[col] = X[col].fillna("NA")
                print(f"Imputed missing values in categorical feature '{col}' with 'NA'.")

        # Re-identify numerical and categorical features after imputation
        # This is important in case dtypes changed or columns were entirely removed/added (not in this case)
        final_numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        final_categorical_features = X.select_dtypes(include="object").columns.tolist()

        # Define the ColumnTransformer for scaling numerical features and encoding categorical features
        # StandardScaler for numerical features
        # OneHotEncoder for categorical features (handle_unknown='ignore' for robustness)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_scaler", StandardScaler(), final_numerical_features),
                (
                    "cat_encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    final_categorical_features,
                ),
            ],
            remainder="drop",  # Drop any columns not explicitly transformed
        )

        # Fit and transform the features
        X_transformed_array = preprocessor.fit_transform(X)

        # Get feature names after one-hot encoding
        # This requires fitting the one-hot encoder and then getting names
        # Extract the one-hot encoder from the column transformer
        ohe = preprocessor.named_transformers_["cat_encoder"]
        feature_names_ohe = ohe.get_feature_names_out(final_categorical_features)

        # Combine all feature names
        transformed_feature_names = final_numerical_features + list(feature_names_ohe)

        # Create a new DataFrame from the transformed array
        X_transformed = pd.DataFrame(X_transformed_array, columns=transformed_feature_names, index=X.index)
        print("Features scaled and encoded.")

        # Recombine with target variable if it was present
        if y is not None:
            transformed_df = pd.concat([X_transformed, y], axis=1)
            print("Transformed features recombined with transformed target variable.")
        else:
            transformed_df = X_transformed
            print("Transformed features returned without target variable.")

        return transformed_df
