from dataclasses import dataclass

import joblib
import mlflow
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

from src.config import (
    RUNTIME_CONFIG,
)
from src.model_bundle import ModelArtifact
from src.preprocessing import (
    DEFAULT_PREPROCESSOR,
    DataPreprocessor,
)


RANDOM_STATE = 42
GRID_SEARCH_JOBS = 1


@dataclass(frozen=True)
class TrainingMetrics:
    cv_best_roc_auc: float
    test_roc_auc: float
    test_accuracy: float


class ChurnModelTrainer:
    def __init__(self, config=RUNTIME_CONFIG, preprocessor: DataPreprocessor | None = None):
        self.config = config
        self.preprocessor = preprocessor or DEFAULT_PREPROCESSOR

    def configure_tracking(self) -> None:
        self.config.ensure_runtime_dirs()
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_registry_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

    def load_training_frame(self):
        raw_df = self.config.load_dataset()
        return self.preprocessor.clean(raw_df)

    def split_training_data(self, df):
        features = df.drop(self.config.target_column, axis=1)
        target = self.preprocessor.encode_target(df[self.config.target_column])
        return train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=target,
        )

    def create_pipeline(self, feature_frame):
        transformer, schema = self.preprocessor.build_preprocessor(feature_frame)
        pipeline = Pipeline(
            [
                ("preprocessor", transformer),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    XGBClassifier(
                        tree_method="hist",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        return pipeline, schema

    def get_param_grid(self):
        return {
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.01, 0.05],
        }

    def build_artifact(self, pipeline, feature_columns, feature_defaults) -> ModelArtifact:
        return ModelArtifact(
            pipeline=pipeline,
            feature_columns=feature_columns,
            numeric_defaults=feature_defaults.numeric_defaults,
            categorical_defaults=feature_defaults.categorical_defaults,
        )

    def save_artifact(self, artifact: ModelArtifact) -> None:
        joblib.dump(artifact.to_payload(), self.config.model_path)
        mlflow.log_artifact(str(self.config.model_path))

    def train(self) -> TrainingMetrics:
        self.configure_tracking()
        df = self.load_training_frame()
        X_train, X_test, y_train, y_test = self.split_training_data(df)
        pipeline, schema = self.create_pipeline(X_train)
        feature_defaults = self.preprocessor.derive_feature_defaults(X_train)

        with mlflow.start_run():
            grid = GridSearchCV(
                pipeline,
                self.get_param_grid(),
                cv=3,
                scoring="roc_auc",
                n_jobs=GRID_SEARCH_JOBS,
                verbose=1,
                error_score="raise",
            )
            grid.fit(X_train, y_train)

            best_pipeline = grid.best_estimator_
            test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
            test_predictions = (
                test_probabilities >= self.config.prediction_threshold
            ).astype(int)

            metrics = TrainingMetrics(
                cv_best_roc_auc=grid.best_score_,
                test_roc_auc=roc_auc_score(y_test, test_probabilities),
                test_accuracy=accuracy_score(y_test, test_predictions),
            )

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("cv_best_roc_auc", metrics.cv_best_roc_auc)
            mlflow.log_metric("test_roc_auc", metrics.test_roc_auc)
            mlflow.log_metric("test_accuracy", metrics.test_accuracy)
            mlflow.log_param("feature_count", X_train.shape[1])
            mlflow.log_param("numeric_feature_count", len(schema.numeric_columns))
            mlflow.log_param(
                "categorical_feature_count", len(schema.categorical_columns)
            )

            artifact = self.build_artifact(
                best_pipeline,
                X_train.columns.tolist(),
                feature_defaults,
            )
            self.save_artifact(artifact)

        print("Saved model to", self.config.model_path)
        print("Best CV ROC AUC:", metrics.cv_best_roc_auc)
        print("Test ROC AUC:", metrics.test_roc_auc)
        print("Test Accuracy:", metrics.test_accuracy)
        return metrics


def main():
    trainer = ChurnModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
