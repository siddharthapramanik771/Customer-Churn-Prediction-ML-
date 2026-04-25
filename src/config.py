from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
POSITIVE_TARGET_LABEL = "Yes"
NEGATIVE_TARGET_LABEL = "No"
MLFLOW_EXPERIMENT_NAME = "churn_prediction"
PREDICTION_THRESHOLD = 0.5
CELERY_BROKER_URL = "redis://redis:6379/0"
CELERY_RESULT_BACKEND = "redis://redis:6379/0"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported dataset format: {path}")


def ensure_runtime_dirs() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)


def get_mlflow_tracking_uri() -> str:
    return MLFLOW_TRACKING_DIR.resolve().as_uri()
