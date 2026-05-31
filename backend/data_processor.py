import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class PreprocessingPipeline:
    def __init__(self, fill_numeric="mean", fill_categorical="mode", scaling="standard"):
        self.fill_numeric = fill_numeric
        self.fill_categorical = fill_categorical
        self.scaling = scaling
        
        # Fitted parameters
        self.impute_values = {}
        self.scaler = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.encoded_cols_map = {} # Maps original categorical column name to the dummy columns generated
        self.all_feature_cols = [] # Final list of feature columns in order
        self.target_col = None

    def profile_data(self, df: pd.DataFrame):
        """Generates statistical profile of a dataframe."""
        profile = []
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_pct = float(missing_count / len(df) * 100)
            
            col_type = "numerical" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
            
            stats = {
                "name": col,
                "type": col_type,
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2)
            }
            
            if col_type == "numerical":
                # Ensure no NaN is passed to JSON
                stats.update({
                    "mean": float(df[col].mean()) if not df[col].isna().all() else 0.0,
                    "std": float(df[col].std()) if not df[col].isna().all() else 0.0,
                    "min": float(df[col].min()) if not df[col].isna().all() else 0.0,
                    "max": float(df[col].max()) if not df[col].isna().all() else 0.0,
                    "median": float(df[col].median()) if not df[col].isna().all() else 0.0,
                })
            else:
                stats.update({
                    "unique_count": int(df[col].nunique()),
                    "top_value": str(df[col].mode().iloc[0]) if not df[col].isna().all() else "None",
                })
            profile.append(stats)
            
        return profile

    def fit_transform(self, df: pd.DataFrame, target_col: str, feature_cols: list) -> tuple:
        """
        Fits the cleaning and scaling pipeline on the features of df and returns (X_processed, y).
        X_processed is a pandas DataFrame.
        """
        self.target_col = target_col
        df = df.copy()
        
        # 1. Fill missing values or drop rows in the target
        if target_col in df.columns:
            df = df.dropna(subset=[target_col])
            y = df[target_col].values
        else:
            y = None

        # Filter dataframe features
        df_features = df[feature_cols].copy()
        
        # Segment numeric and categorical columns
        self.numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_features[c])]
        self.categorical_cols = [c for c in feature_cols if c not in self.numeric_cols]
        
        # 2. Impute Numeric Features
        for col in self.numeric_cols:
            if df_features[col].isna().sum() > 0:
                if self.fill_numeric == "mean":
                    val = df_features[col].mean()
                elif self.fill_numeric == "median":
                    val = df_features[col].median()
                elif self.fill_numeric == "mode":
                    val = df_features[col].mode().iloc[0] if not df_features[col].mode().empty else 0.0
                else:
                    val = 0.0 # fallback constant
                
                self.impute_values[col] = float(val)
                df_features[col] = df_features[col].fillna(val)
            else:
                self.impute_values[col] = 0.0

        # 3. Impute Categorical Features
        for col in self.categorical_cols:
            if df_features[col].isna().sum() > 0:
                if self.fill_categorical == "mode":
                    val = df_features[col].mode().iloc[0] if not df_features[col].mode().empty else "Unknown"
                else:
                    val = "Unknown"
                
                self.impute_values[col] = str(val)
                df_features[col] = df_features[col].fillna(val)
            else:
                self.impute_values[col] = "Unknown"

        # Convert categorical cols to string
        for col in self.categorical_cols:
            df_features[col] = df_features[col].astype(str)

        # 4. Handle Categorical Encoding (One-Hot Encoding)
        # We manually perform one-hot encoding to strictly track category names
        X_parts = []
        
        # Add numeric part
        if self.numeric_cols:
            X_parts.append(df_features[self.numeric_cols].reset_index(drop=True))
            
        # Add one-hot encoded categorical columns
        for col in self.categorical_cols:
            # Learn categories
            cats = sorted(list(df_features[col].unique()))
            self.encoded_cols_map[col] = cats
            
            # Create dummy columns manually so we are 100% consistent during transform()
            dummies = pd.DataFrame(0, index=range(len(df_features)), columns=[f"{col}_{cat}" for cat in cats])
            for i, val in enumerate(df_features[col]):
                if f"{col}_{val}" in dummies.columns:
                    dummies.loc[i, f"{col}_{val}"] = 1
            X_parts.append(dummies)
            
        # Combine parts
        if X_parts:
            X_processed = pd.concat(X_parts, axis=1)
        else:
            X_processed = pd.DataFrame()
            
        self.all_feature_cols = list(X_processed.columns)

        # 5. Fit & Transform Scaling (Numeric only)
        if self.numeric_cols and self.scaling in ["standard", "minmax"]:
            if self.scaling == "standard":
                self.scaler = StandardScaler()
            elif self.scaling == "minmax":
                self.scaler = MinMaxScaler()
                
            # Scale numeric features in-place inside X_processed
            X_processed[self.numeric_cols] = self.scaler.fit_transform(X_processed[self.numeric_cols])

        return X_processed, y

    def transform(self, input_dict: dict) -> np.ndarray:
        """
        Transforms a single row input dictionary using the fitted pipeline parameters.
        Returns a 2D numpy array ready for model predict: shape (1, num_features).
        """
        if not self.all_feature_cols:
            raise ValueError("Pipeline has not been fitted yet!")

        # Create empty row dataframe mapping final features
        df_row = pd.DataFrame(0.0, index=[0], columns=self.all_feature_cols)

        # 1. Fill Numerical Values
        for col in self.numeric_cols:
            val = input_dict.get(col)
            # Impute if missing or None
            if val is None or val == "":
                val = self.impute_values.get(col, 0.0)
            else:
                try:
                    val = float(val)
                except ValueError:
                    val = self.impute_values.get(col, 0.0)
            
            df_row.loc[0, col] = val

        # 2. Scale Numerical Features
        if self.numeric_cols and self.scaler is not None:
            # We scale only the numeric columns inside our row
            df_row[self.numeric_cols] = self.scaler.transform(df_row[self.numeric_cols])

        # 3. Fill and Encode Categorical Values
        for col in self.categorical_cols:
            val = input_dict.get(col)
            if val is None or val == "":
                val = self.impute_values.get(col, "Unknown")
            else:
                val = str(val)

            # Set one-hot category column
            dummy_col = f"{col}_{val}"
            if dummy_col in df_row.columns:
                df_row.loc[0, dummy_col] = 1.0
            else:
                # If we encounter an unseen category, all dummy columns for this feature remain 0
                pass

        return df_row.values
