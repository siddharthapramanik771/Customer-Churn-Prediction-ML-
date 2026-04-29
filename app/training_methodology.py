import streamlit as st

from src.config import RUNTIME_CONFIG, RuntimeConfig


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
