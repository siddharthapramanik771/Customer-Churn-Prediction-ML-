import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    return df

def build_preprocessor(df: pd.DataFrame):
    # simple heuristic split; adapt as needed
    num_cols = [c for c in df.columns if df[c].dtype != 'object' and c != 'churn']
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']

    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    return pre, num_cols, cat_cols
