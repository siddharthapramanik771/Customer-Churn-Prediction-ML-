import joblib
import numpy as np

BUNDLE_PATH = "models/model.joblib"

bundle = joblib.load(BUNDLE_PATH)
pre = bundle["preprocessor"]
model = bundle["model"]

def predict_proba_single(payload: dict):
    import pandas as pd
    X = pd.DataFrame([payload])
    X_enc = pre.transform(X)
    proba = model.predict_proba(X_enc)[0,1]
    return float(proba)
