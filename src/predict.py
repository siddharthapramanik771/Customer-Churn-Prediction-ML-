import joblib
from src.config import MODEL_PATH

bundle = joblib.load(MODEL_PATH)
pre = bundle["preprocessor"]
model = bundle["model"]

def predict_proba_single(payload: dict):
    import pandas as pd
    import numpy as np

    X = pd.DataFrame([payload])
    # Ensure all expected columns are present. The fitted ColumnTransformer
    # stores which columns it operates on in `transformers_`. We'll collect
    # those column names and fill missing numeric columns with 0.0 and
    # categorical columns with a placeholder string so `transform` won't
    # fail on single-row partial payloads.
    required_cols = []
    col_kind = {}
    for name, transformer, cols in getattr(pre, "transformers_", []):
        # skip passthrough/remainder entries that don't list columns
        if cols in ("remainder", None):
            continue
        # cols can be list of names, slice, or array of indices
        if isinstance(cols, (list, tuple)):
            for c in cols:
                if isinstance(c, str):
                    required_cols.append(c)
                    col_kind[c] = name
        elif isinstance(cols, slice):
            if hasattr(pre, "feature_names_in_"):
                names = list(pre.feature_names_in_[cols])
                for c in names:
                    required_cols.append(c)
                    col_kind[c] = name
        else:
            # numpy array of indices or other; try mapping via feature_names_in_
            try:
                import numpy as _np
                if isinstance(cols, _np.ndarray) and hasattr(pre, "feature_names_in_"):
                    for i in cols.tolist():
                        c = pre.feature_names_in_[int(i)]
                        required_cols.append(c)
                        col_kind[c] = name
            except Exception:
                pass

    # Deduplicate while preserving order
    seen = set()
    req = []
    for c in required_cols:
        if c not in seen:
            seen.add(c)
            req.append(c)

    # Fill missing columns with sensible defaults
    for c in req:
        if c not in X.columns:
            if col_kind.get(c) == "num":
                X[c] = 0.0
            else:
                X[c] = "missing"

    if req:
        X = X.reindex(columns=req)

    X_enc = pre.transform(X)
    proba = model.predict_proba(X_enc)[0,1]
    return float(proba)
