from dataclasses import dataclass
import json

import altair as alt
import pandas as pd
import streamlit as st

from src.config import RUNTIME_CONFIG, RuntimeConfig


@dataclass(frozen=True)
class HoldoutMetrics:
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]


class TrainingMethodologyRenderer:
    """Renders a readable explanation of the model training workflow."""

    def __init__(self, config: RuntimeConfig = RUNTIME_CONFIG) -> None:
        self.config = config

    def render(self) -> None:
        st.header("Training Methodology")
        st.write(
            "The churn model is trained as a supervised binary classifier. "
            "Historical customer records are cleaned, transformed, balanced, "
            "and used to train an XGBoost model that predicts churn probability."
        )

        self.render_workflow()
        self.render_pipeline()
        self.render_validation()
        self.render_test_metrics()
        self.render_artifact()

    def render_workflow(self) -> None:
        st.subheader("Workflow")
        steps = [
            "Load the customer dataset from `data/data.csv`.",
            "Clean column names, string values, missing values, and numeric-looking text.",
            f"Remove the identifier column `{self.config.id_column}` before training.",
            f"Encode `{self.config.target_column}` as a binary target.",
            "Split the data into training and test sets with stratification.",
            "Fit preprocessing, SMOTE balancing, and XGBoost inside one pipeline.",
            "Use grid search cross-validation to choose model hyperparameters.",
            "Evaluate the best model on the holdout test set.",
            "Save the trained pipeline and prediction metadata to `models/model.joblib`.",
        ]

        for index, step in enumerate(steps, start=1):
            st.write(f"{index}. {step}")

    def render_pipeline(self) -> None:
        st.subheader("Model Pipeline")

        st.markdown(
            """
| Stage | Method | Purpose |
| --- | --- | --- |
| Cleaning | `DataPreprocessor.clean` | Normalize raw data and remove unusable rows |
| Numeric features | `StandardScaler` | Standardize numeric inputs |
| Categorical features | `OneHotEncoder(handle_unknown="ignore")` | Convert categories into model-ready columns |
| Class balancing | `SMOTE` | Add synthetic minority-class samples during training |
| Classifier | `XGBClassifier` | Train a gradient-boosted tree model |
"""
        )

        st.info(
            "SMOTE is inside the training pipeline, so it is applied only to "
            "training folds during cross-validation. The final test data remains untouched."
        )

    def render_validation(self) -> None:
        st.subheader("Validation Strategy")

        columns = st.columns(3)
        columns[0].metric("Test split", "20%")
        columns[1].metric("CV folds", "3")
        columns[2].metric("Selection metric", "ROC AUC")

        st.write(
            "The train/test split is stratified to preserve the churn/non-churn "
            "ratio in both sets. Grid search evaluates combinations of "
            "`n_estimators`, `max_depth`, and `learning_rate`, then keeps the "
            "pipeline with the best cross-validated ROC AUC."
        )

        st.markdown(
            """
```python
{
    "model__n_estimators": [100, 200],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.01, 0.05],
}
```
"""
        )

    def render_test_metrics(self) -> None:
        st.subheader("Current Test Metrics")

        try:
            metrics = self.load_saved_metrics()
        except Exception as exc:
            st.warning(f"Test metrics are unavailable: {exc}")
            return

        metric_columns = st.columns(5)
        metric_columns[0].metric("Accuracy", f"{metrics.accuracy:.2%}")
        metric_columns[1].metric("ROC AUC", f"{metrics.roc_auc:.3f}")
        metric_columns[2].metric("Precision", f"{metrics.precision:.2%}")
        metric_columns[3].metric("Recall", f"{metrics.recall:.2%}")
        metric_columns[4].metric("F1 score", f"{metrics.f1:.3f}")

        st.write(
            "These scores were calculated during training and saved as a small "
            "metrics artifact. The dashboard reads the stored report instead of "
            "re-evaluating the model at page load."
        )

        left, right = st.columns(2)
        left.altair_chart(
            self.build_metric_chart(metrics),
            use_container_width=True,
        )
        right.altair_chart(
            self.build_confusion_matrix_chart(metrics),
            use_container_width=True,
        )

    def load_saved_metrics(self) -> HoldoutMetrics:
        if not self.config.metrics_path.exists():
            raise FileNotFoundError(
                f"Metrics artifact not found at {self.config.metrics_path}. "
                "Run training to create it."
            )

        payload = json.loads(self.config.metrics_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        return HoldoutMetrics(
            accuracy=metrics["test_accuracy"],
            roc_auc=metrics["test_roc_auc"],
            precision=metrics["test_precision"],
            recall=metrics["test_recall"],
            f1=metrics["test_f1"],
            confusion=metrics["confusion_matrix"],
        )

    @staticmethod
    def build_metric_chart(metrics: HoldoutMetrics) -> alt.Chart:
        metric_df = pd.DataFrame(
            [
                {"metric": "Accuracy", "score": metrics.accuracy},
                {"metric": "ROC AUC", "score": metrics.roc_auc},
                {"metric": "Precision", "score": metrics.precision},
                {"metric": "Recall", "score": metrics.recall},
                {"metric": "F1", "score": metrics.f1},
            ]
        )

        return (
            alt.Chart(metric_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("score:Q", legend=None),
                tooltip=["metric:N", alt.Tooltip("score:Q", format=".3f")],
            )
            .properties(height=300)
        )

    def build_confusion_matrix_chart(self, metrics: HoldoutMetrics) -> alt.Chart:
        labels = [
            self.config.negative_target_label,
            self.config.positive_target_label,
        ]
        matrix_rows = []
        for actual_index, actual_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                matrix_rows.append(
                    {
                        "actual": actual_label,
                        "predicted": predicted_label,
                        "customers": metrics.confusion[actual_index][predicted_index],
                    }
                )

        matrix_df = pd.DataFrame(matrix_rows)
        heatmap = (
            alt.Chart(matrix_df)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", title="Predicted"),
                y=alt.Y("actual:N", title="Actual"),
                color=alt.Color("customers:Q", title="Customers"),
                tooltip=["actual:N", "predicted:N", "customers:Q"],
            )
        )
        labels_chart = (
            alt.Chart(matrix_df)
            .mark_text(fontSize=18, fontWeight="bold")
            .encode(
                x="predicted:N",
                y="actual:N",
                text="customers:Q",
                color=alt.value("white"),
            )
        )
        return (heatmap + labels_chart).properties(height=300)

    def render_artifact(self) -> None:
        st.subheader("Saved Model Artifact")
        st.write(
            "Training saves a unified model bundle containing the fitted pipeline, "
            "the training-time feature order, default values for missing inputs, "
            "target labels, and the prediction threshold."
        )

        st.markdown(
            f"""
| Artifact Field | Value |
| --- | --- |
| Model path | `{self.config.model_path}` |
| Metrics path | `{self.config.metrics_path}` |
| Target column | `{self.config.target_column}` |
| Positive label | `{self.config.positive_target_label}` |
| Negative label | `{self.config.negative_target_label}` |
| Prediction threshold | `{self.config.prediction_threshold}` |
"""
        )

        st.write(
            "The dashboard uses this saved artifact for prediction, so the deployed "
            "app does not retrain the model while users interact with it."
        )
