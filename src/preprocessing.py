import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ID_COLUMN,
    NEGATIVE_TARGET_LABEL,
    POSITIVE_TARGET_LABEL,
    TARGET_COLUMN,
)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
    return df

def build_preprocessor(df: pd.DataFrame):
    # simple heuristic split; adapt as needed
    feature_df = df.drop(columns=[TARGET_COLUMN], errors='ignore')
    num_cols = [c for c in feature_df.columns if feature_df[c].dtype != 'object']
    cat_cols = [c for c in feature_df.columns if feature_df[c].dtype == 'object']

    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    return pre, num_cols, cat_cols


def encode_target(target: pd.Series) -> pd.Series:
    normalized = target.astype(str).str.strip()
    mapping = {
        NEGATIVE_TARGET_LABEL: 0,
        POSITIVE_TARGET_LABEL: 1,
    }
    encoded = normalized.map(mapping)

    if encoded.isna().any():
        invalid_values = sorted(normalized[encoded.isna()].unique().tolist())
        raise ValueError(
            f"Unexpected values in target column '{TARGET_COLUMN}': {invalid_values}. "
            f"Expected only {NEGATIVE_TARGET_LABEL!r} and {POSITIVE_TARGET_LABEL!r}."
        )

    return encoded.astype(int)
