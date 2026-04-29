from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RuntimeConfig:
    """Application settings shared by training, prediction, and the UI."""

    project_root: Path
    data_path: Path
    model_path: Path
    metrics_path: Path
    mlflow_tracking_dir: Path
    target_column: str
    id_column: str
    positive_target_label: str
    negative_target_label: str
    mlflow_experiment_name: str
    prediction_threshold: float

    @classmethod
    def from_project_root(cls, project_root: Path) -> "RuntimeConfig":
        return cls(
            project_root=project_root,
            data_path=project_root / "data" / "data.csv",
            model_path=project_root / "models" / "model.joblib",
            metrics_path=project_root / "models" / "training_metrics.json",
            mlflow_tracking_dir=project_root / "mlruns",
            target_column="Churn",
            id_column="customerID",
            positive_target_label="Yes",
            negative_target_label="No",
            mlflow_experiment_name="churn_prediction",
            prediction_threshold=0.5,
        )

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
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracking_dir.mkdir(parents=True, exist_ok=True)

    @property
    def mlflow_tracking_uri(self) -> str:
        return self.mlflow_tracking_dir.resolve().as_uri()


RUNTIME_CONFIG = RuntimeConfig.from_project_root(
    Path(__file__).resolve().parent.parent
)
