from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path
    data_path: Path
    model_path: Path
    mlflow_tracking_dir: Path
    target_column: str
    id_column: str
    positive_target_label: str
    negative_target_label: str
    mlflow_experiment_name: str
    prediction_threshold: float
    celery_broker_url: str
    celery_result_backend: str

    def load_dataset(self, path: Path | None = None) -> pd.DataFrame:
        dataset_path = path or self.data_path
        suffix = dataset_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(dataset_path)

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(dataset_path)

        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    def ensure_runtime_dirs(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracking_dir.mkdir(parents=True, exist_ok=True)

    @property
    def mlflow_tracking_uri(self) -> str:
        return self.mlflow_tracking_dir.resolve().as_uri()


RUNTIME_CONFIG = RuntimeConfig(
    project_root=Path(__file__).resolve().parent.parent,
    data_path=Path(__file__).resolve().parent.parent / "data" / "data.csv",
    model_path=Path(__file__).resolve().parent.parent / "models" / "model.joblib",
    mlflow_tracking_dir=Path(__file__).resolve().parent.parent / "mlruns",
    target_column="Churn",
    id_column="customerID",
    positive_target_label="Yes",
    negative_target_label="No",
    mlflow_experiment_name="churn_prediction",
    prediction_threshold=0.5,
    celery_broker_url="redis://redis:6379/0",
    celery_result_backend="redis://redis:6379/0",
)

PROJECT_ROOT = RUNTIME_CONFIG.project_root
DATA_PATH = RUNTIME_CONFIG.data_path
MODEL_PATH = RUNTIME_CONFIG.model_path
MLFLOW_TRACKING_DIR = RUNTIME_CONFIG.mlflow_tracking_dir
TARGET_COLUMN = RUNTIME_CONFIG.target_column
ID_COLUMN = RUNTIME_CONFIG.id_column
POSITIVE_TARGET_LABEL = RUNTIME_CONFIG.positive_target_label
NEGATIVE_TARGET_LABEL = RUNTIME_CONFIG.negative_target_label
MLFLOW_EXPERIMENT_NAME = RUNTIME_CONFIG.mlflow_experiment_name
PREDICTION_THRESHOLD = RUNTIME_CONFIG.prediction_threshold
CELERY_BROKER_URL = RUNTIME_CONFIG.celery_broker_url
CELERY_RESULT_BACKEND = RUNTIME_CONFIG.celery_result_backend


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    return RUNTIME_CONFIG.load_dataset(path)


def ensure_runtime_dirs() -> None:
    RUNTIME_CONFIG.ensure_runtime_dirs()


def get_mlflow_tracking_uri() -> str:
    return RUNTIME_CONFIG.mlflow_tracking_uri
