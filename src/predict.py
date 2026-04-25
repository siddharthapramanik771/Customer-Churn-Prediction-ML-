import joblib
from config import MODEL_PATH

bundle = joblib.load(MODEL_PATH)
pre = bundle["preprocessor"]
model = bundle["model"]

def predict_proba_single(payload: dict):
    import pandas as pd
    X = pd.DataFrame([payload])
    X_enc = pre.transform(X)
    proba = model.predict_proba(X_enc)[0,1]
    return float(proba)
