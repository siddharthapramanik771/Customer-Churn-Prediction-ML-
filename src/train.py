import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.model_bundle import ModelArtifact, ModelArtifactRepository
from src.preprocessing import (
    DataPreprocessor,
    FeatureDefaults,
    FeatureSchema,
)


@dataclass(frozen=True)
class TrainingSettings:
    """Settings that control reproducibility, validation, and search behavior."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 3
    scoring: str = "roc_auc"
    grid_search_jobs: int = 1
    grid_search_verbose: int = 1
    param_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.01, 0.05],
        }
    )


@dataclass(frozen=True)
class TrainingMetrics:
    cv_best_roc_auc: float
    test_roc_auc: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    confusion_matrix: list[list[int]]
    test_size: float
    random_state: int
    cv_folds: int
    scoring: str

    def to_log_dict(self) -> dict[str, float]:
        return {
            "cv_best_roc_auc": self.cv_best_roc_auc,
            "test_roc_auc": self.test_roc_auc,
            "test_accuracy": self.test_accuracy,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "test_f1": self.test_f1,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            **self.to_log_dict(),
            "confusion_matrix": self.confusion_matrix,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "cv_folds": self.cv_folds,
            "scoring": self.scoring,
        }


class MLflowTrainingTracker:
    """Small adapter around MLflow so tracking stays outside model logic."""

    def __init__(self, config: RuntimeConfig = RUNTIME_CONFIG) -> None:
        self.config = config

    def configure(self) -> None:
        self.config.ensure_runtime_dirs()
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_registry_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

    def log_run(
        self,
        *,
        best_params: dict[str, Any],
        metrics: TrainingMetrics,
        schema: FeatureSchema,
        feature_count: int,
        artifact_path: str,
        metrics_path: str,
    ) -> None:
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics.to_log_dict())
        mlflow.log_param("feature_count", feature_count)
        mlflow.log_param("numeric_feature_count", len(schema.numeric_columns))
        mlflow.log_param("categorical_feature_count", len(schema.categorical_columns))
        mlflow.log_artifact(artifact_path)
        mlflow.log_artifact(metrics_path)


class ChurnModelTrainer:
    """Coordinates the end-to-end model training workflow."""

    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        preprocessor: DataPreprocessor | None = None,
        settings: TrainingSettings | None = None,
        artifact_repository: ModelArtifactRepository | None = None,
        tracker: MLflowTrainingTracker | None = None,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor or DataPreprocessor.from_config(config)
        self.settings = settings or TrainingSettings()
        self.artifact_repository = artifact_repository or ModelArtifactRepository(
            config.model_path
        )
        self.tracker = tracker or MLflowTrainingTracker(config)

    def train(self) -> TrainingMetrics:
        self.tracker.configure()
        df = self.load_training_frame()
        X_train, X_test, y_train, y_test = self.split_training_data(df)
        pipeline, schema = self.create_pipeline(X_train)
        feature_defaults = self.preprocessor.derive_feature_defaults(X_train)

        with mlflow.start_run():
            grid = self.create_grid_search(pipeline)
            grid.fit(X_train, y_train)

            best_pipeline = grid.best_estimator_
            test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
            test_predictions = (
                test_probabilities >= self.config.prediction_threshold
            ).astype(int)

            metrics = self.evaluate(
                grid.best_score_,
                y_test,
                test_probabilities,
                test_predictions,
            )
            artifact = self.build_artifact(
                best_pipeline,
                X_train.columns.tolist(),
                feature_defaults,
            )
            self.artifact_repository.save(artifact)
            self.save_metrics_artifact(metrics, grid.best_params_)
            self.tracker.log_run(
                best_params=grid.best_params_,
                metrics=metrics,
                schema=schema,
                feature_count=X_train.shape[1],
                artifact_path=str(self.artifact_repository.model_path),
                metrics_path=str(self.config.metrics_path),
            )

        self.print_summary(metrics)
        return metrics

    def load_training_frame(self) -> pd.DataFrame:
        raw_df = self.config.load_dataset()
        return self.preprocessor.clean(raw_df)

    def split_training_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        features = df.drop(self.config.target_column, axis=1)
        target = self.preprocessor.encode_target(df[self.config.target_column])
        return train_test_split(
            features,
            target,
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
            stratify=target,
        )

    def create_pipeline(
        self, feature_frame: pd.DataFrame
    ) -> tuple[Pipeline, FeatureSchema]:
        transformer, schema = self.preprocessor.build_transformer(feature_frame)
        pipeline = Pipeline(
            [
                ("preprocessor", transformer),
                ("smote", SMOTE(random_state=self.settings.random_state)),
                (
                    "model",
                    XGBClassifier(
                        tree_method="hist",
                        eval_metric="logloss",
                        random_state=self.settings.random_state,
                    ),
                ),
            ]
        )
        return pipeline, schema

    def create_grid_search(self, pipeline: Pipeline) -> GridSearchCV:
        return GridSearchCV(
            pipeline,
            self.settings.param_grid,
            cv=self.settings.cv_folds,
            scoring=self.settings.scoring,
            n_jobs=self.settings.grid_search_jobs,
            verbose=self.settings.grid_search_verbose,
            error_score="raise",
        )

    def evaluate(
        self,
        cv_best_score: float,
        y_test: pd.Series,
        test_probabilities,
        test_predictions,
    ) -> TrainingMetrics:
        return TrainingMetrics(
            cv_best_roc_auc=cv_best_score,
            test_roc_auc=roc_auc_score(y_test, test_probabilities),
            test_accuracy=accuracy_score(y_test, test_predictions),
            test_precision=precision_score(y_test, test_predictions, zero_division=0),
            test_recall=recall_score(y_test, test_predictions, zero_division=0),
            test_f1=f1_score(y_test, test_predictions, zero_division=0),
            confusion_matrix=confusion_matrix(y_test, test_predictions).tolist(),
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
            cv_folds=self.settings.cv_folds,
            scoring=self.settings.scoring,
        )

    def save_metrics_artifact(
        self, metrics: TrainingMetrics, best_params: dict[str, Any]
    ) -> None:
        payload = {
            "metrics": metrics.to_artifact_dict(),
            "best_params": best_params,
            "training_data_path": self.relative_project_path(self.config.data_path),
            "training_data_name": self.config.data_path.name,
            "target_column": self.config.target_column,
            "positive_target_label": self.config.positive_target_label,
            "negative_target_label": self.config.negative_target_label,
            "prediction_threshold": self.config.prediction_threshold,
        }
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metrics_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def relative_project_path(self, path: Path) -> str:
        resolved_path = path.resolve()
        try:
            return resolved_path.relative_to(self.config.project_root.resolve()).as_posix()
        except ValueError:
            return str(resolved_path)

    def build_artifact(
        self,
        pipeline: Pipeline,
        feature_columns: list[str],
        feature_defaults: FeatureDefaults,
    ) -> ModelArtifact:
        return ModelArtifact(
            pipeline=pipeline,
            feature_columns=feature_columns,
            numeric_defaults=feature_defaults.numeric_defaults,
            categorical_defaults=feature_defaults.categorical_defaults,
            target_column=self.config.target_column,
            positive_target_label=self.config.positive_target_label,
            negative_target_label=self.config.negative_target_label,
            prediction_threshold=self.config.prediction_threshold,
        )

    def print_summary(self, metrics: TrainingMetrics) -> None:
        print("Saved model to", self.artifact_repository.model_path)
        print("Saved metrics to", self.config.metrics_path)
        print("Best CV ROC AUC:", metrics.cv_best_roc_auc)
        print("Test ROC AUC:", metrics.test_roc_auc)
        print("Test Accuracy:", metrics.test_accuracy)
        print("Test Precision:", metrics.test_precision)
        print("Test Recall:", metrics.test_recall)
        print("Test F1:", metrics.test_f1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the customer churn model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to a CSV/XLS/XLSX training dataset. Defaults to data/data.csv.",
    )
    return parser.parse_args()


def resolve_data_path(data_path: Path | None, config: RuntimeConfig) -> Path | None:
    if data_path is None:
        return None
    if data_path.is_absolute():
        return data_path
    return config.project_root / data_path


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data, RUNTIME_CONFIG)
    config = RUNTIME_CONFIG.with_data_path(data_path) if data_path else RUNTIME_CONFIG
    trainer = ChurnModelTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
