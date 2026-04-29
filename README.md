# Customer Churn Prediction

## Overview

This repository contains an end-to-end customer churn prediction system. It
includes data preprocessing, model training, experiment tracking, model artifact
persistence, local prediction, and a Streamlit dashboard for interactive scoring
and dataset exploration.

The runtime application is intentionally simple: Streamlit loads a trained model
artifact and performs prediction in-process. Training remains a separate offline
workflow.

```text
raw customer data
  -> preprocessing
  -> train/test split
  -> feature transformation
  -> class balancing
  -> model training
  -> evaluation
  -> artifact persistence
  -> Streamlit prediction
```

## Problem Definition

Customer churn refers to a customer leaving, canceling, or stopping use of a
service. The system predicts whether a customer is likely to churn from customer
profile, service, billing, and usage features.

The machine learning task is supervised binary classification:

- supervised: historical rows include the known target value
- binary: the target has two possible labels, `Yes` and `No`
- classification: the final output is a class label

The model also produces a churn probability. The probability is converted into a
final label using a configurable threshold.

```text
probability >= 0.5 -> Yes
probability < 0.5  -> No
```

## Repository Layout

```text
.
|-- app/
|   `-- app.py                    # Streamlit UI and app-facing services
|-- data/
|   |-- data.csv                  # Historical customer dataset
|   `-- README.md                 # Dataset placement note
|-- models/
|   `-- model.joblib              # Trained model artifact
|-- notebooks/
|   `-- train_churn_model.ipynb   # Step-by-step training notebook
|-- src/
|   |-- config.py                 # Runtime settings
|   |-- model_bundle.py           # Model artifact contract and persistence
|   |-- predict.py                # Prediction services
|   |-- preprocessing.py          # Cleaning and feature preparation
|   `-- train.py                  # Offline training workflow
|-- docker-compose.yml            # Local container orchestration
|-- Dockerfile.streamlit          # Streamlit runtime image
|-- requirements.txt              # Runtime dependencies
|-- requirements-train.txt        # Training and notebook dependencies
`-- streamlit_app.py              # Streamlit entrypoint
```

## Runtime Architecture

The project follows a small service-oriented object model. Each class owns a
single responsibility and receives its dependencies explicitly where practical.

| Component | File | Responsibility |
| --- | --- | --- |
| `RuntimeConfig` | `src/config.py` | Central paths, labels, threshold, and MLflow settings |
| `DataPreprocessor` | `src/preprocessing.py` | Data cleaning, schema inference, target encoding, defaults |
| `FeatureSchema` | `src/preprocessing.py` | Numeric and categorical feature groups |
| `FeatureDefaults` | `src/preprocessing.py` | Prediction-time fallback values |
| `TrainingSettings` | `src/train.py` | Train/test split, CV folds, scoring, and hyperparameter grid |
| `TrainingMetrics` | `src/train.py` | Cross-validation and test metrics |
| `MLflowTrainingTracker` | `src/train.py` | Experiment tracking through MLflow |
| `ChurnModelTrainer` | `src/train.py` | End-to-end training orchestration |
| `ModelArtifact` | `src/model_bundle.py` | Serializable model artifact structure |
| `ModelArtifactRepository` | `src/model_bundle.py` | Artifact load/save through `joblib` |
| `FeaturePayloadBuilder` | `src/predict.py` | Payload alignment and missing-value fallback |
| `ChurnPredictor` | `src/predict.py` | Artifact-backed prediction service |
| `ChurnPrediction` | `src/predict.py` | Prediction response object |
| `ReferenceDataService` | `app/app.py` | Cleaned dataset loading for UI reference data |
| `LocalPredictionService` | `app/app.py` | Streamlit-facing prediction wrapper |
| `DashboardRenderer` | `app/app.py` | Streamlit layout, controls, and visual output |

### Design Rationale

The design separates responsibilities that change for different reasons:

- model training can change without changing the Streamlit renderer
- preprocessing rules can evolve independently from model persistence
- artifact storage is isolated in a repository class
- prediction payload alignment is isolated from the UI
- training settings are grouped in a dedicated configuration object
- MLflow logic is kept outside core model-building methods

This structure keeps the project easier to test, debug, extend, and deploy.

## Data Contract

Default dataset path:

```text
data/data.csv
```

Required target column:

```text
Churn
```

Configured target labels:

```text
positive label: Yes
negative label: No
```

Configured ID column:

```text
customerID
```

The ID column is removed before training because it identifies rows rather than
representing customer behavior. Identifier columns can encourage memorization or
spurious patterns and should generally not be used as predictive features.

## Configuration

Configuration is centralized in `src/config.py`.

```python
RUNTIME_CONFIG = RuntimeConfig.from_project_root(...)
```

`RuntimeConfig` stores:

- project root
- dataset path
- model artifact path
- MLflow tracking directory
- target column
- ID column
- positive and negative target labels
- MLflow experiment name
- prediction threshold

Centralized configuration avoids scattered constants and keeps dataset, artifact,
and label changes localized.

## Preprocessing

Preprocessing is implemented by `DataPreprocessor` in `src/preprocessing.py`.

### Cleaning Behavior

`DataPreprocessor.clean` performs the following operations:

1. Copies the input dataframe.
2. Strips whitespace from column names.
3. Normalizes object/string columns.
4. Strips whitespace from string values.
5. Converts blank strings to missing values.
6. Converts numeric-looking text columns to numeric dtype.
7. Drops rows with missing values.
8. Removes the configured ID column.

### Numeric-Looking Text Conversion

CSV files can store numeric values as strings. The preprocessor attempts numeric
conversion for object/string columns and converts a column when at least 95% of
non-missing values are numeric.

```python
NUMERIC_CONVERSION_THRESHOLD = 0.95
```

The threshold permits small amounts of noisy raw data while still recovering
columns that should be numeric.

### Missing Rows

Rows with missing values are currently dropped with `dropna()`. This keeps the
training workflow simple and deterministic. A production workflow with heavier
missingness could replace this with explicit imputation.

## Feature Schema

`DataPreprocessor.infer_schema` separates input features into:

- numeric features
- categorical features

Numeric columns are detected with pandas dtype checks. Categorical columns are
all remaining non-numeric columns.

The inferred schema is stored in `FeatureSchema`:

```python
FeatureSchema(
    numeric_columns=[...],
    categorical_columns=[...],
)
```

## Target Encoding

The target column is converted from text labels into numeric classes:

```text
No  -> 0
Yes -> 1
```

This mapping is performed by `DataPreprocessor.encode_target`.

The method validates the target values and raises an error if unexpected labels
are present. This prevents silent training on malformed targets.

## Feature Transformation

Feature transformation is handled through a scikit-learn `ColumnTransformer`.

```text
numeric columns     -> StandardScaler
categorical columns -> OneHotEncoder
```

### StandardScaler

`StandardScaler` normalizes numeric features by subtracting the mean and dividing
by the standard deviation.

```text
scaled_value = (value - mean) / standard_deviation
```

Tree-based models such as XGBoost do not strictly require scaling, but keeping
numeric preprocessing explicit makes the pipeline consistent and easier to
extend.

### OneHotEncoder

`OneHotEncoder` converts categorical values into binary indicator columns.

Example:

```text
Contract = Month-to-month
Contract = One year
Contract = Two year
```

becomes:

```text
Contract_Month-to-month  Contract_One year  Contract_Two year
1                        0                  0
0                        1                  0
0                        0                  1
```

The encoder is configured with:

```python
OneHotEncoder(handle_unknown="ignore")
```

This prevents prediction failures when app-time input contains a category not
seen during training.

## Train/Test Split

Training uses an 80/20 split:

```python
test_size = 0.2
```

The split is stratified:

```python
stratify=target
```

Stratification preserves the churn/non-churn ratio across train and test sets.
This matters for classification problems where one class may be more common than
the other.

The random seed is controlled by `TrainingSettings.random_state` for repeatable
splits and model behavior.

## Class Imbalance

Churn datasets commonly contain more non-churn customers than churn customers.
This can make accuracy misleading. For example, if most customers do not churn,
a model can achieve high accuracy by overpredicting the majority class.

The project uses SMOTE to reduce this issue.

## SMOTE

SMOTE stands for Synthetic Minority Over-sampling Technique.

It creates synthetic examples of the minority class during training. In this
project, that usually means creating additional churn-like examples.

SMOTE is placed inside the imbalanced-learn pipeline:

```text
ColumnTransformer -> SMOTE -> XGBClassifier
```

This placement is important because cross-validation must apply SMOTE only to
training folds. The final test set must remain untouched to preserve an honest
evaluation.

## Model

The classifier is:

```python
XGBClassifier
```

XGBoost is a gradient boosted decision tree model. A decision tree learns rules
such as:

```text
Is tenure low?
Is the contract month-to-month?
Are customer service calls high?
```

Gradient boosting trains many trees sequentially. Each new tree focuses on
correcting previous errors. XGBoost is commonly effective for structured tabular
datasets.

Current model settings include:

```python
XGBClassifier(
    tree_method="hist",
    eval_metric="logloss",
    random_state=settings.random_state,
)
```

## Training Pipeline

The full pipeline is:

```text
ColumnTransformer -> SMOTE -> XGBClassifier
```

The implementation uses `imblearn.pipeline.Pipeline` because SMOTE changes the
number of rows during training. A plain scikit-learn pipeline is not appropriate
for samplers.

Pipeline benefits:

- one fitted object contains preprocessing and model steps
- cross-validation applies preprocessing correctly per fold
- SMOTE is applied only during training
- prediction uses the same transformations as training
- the entire workflow can be saved as one artifact

## Hyperparameter Search

Hyperparameters are model settings selected before training. The project uses
`GridSearchCV` to evaluate combinations of:

- `n_estimators`
- `max_depth`
- `learning_rate`

Current grid:

```python
{
    "model__n_estimators": [100, 200],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.01, 0.05],
}
```

The `model__` prefix addresses the `model` step inside the pipeline.

## Cross-Validation

Training uses 3-fold cross-validation:

```python
cv_folds = 3
```

Each hyperparameter combination is evaluated across multiple validation folds.
This gives a more stable model-selection signal than a single split.

The scoring metric is:

```python
scoring = "roc_auc"
```

ROC AUC is appropriate for churn classification because it evaluates probability
ranking and is more informative than accuracy alone for imbalanced data.

## Evaluation Metrics

The project reports several metrics.

### ROC AUC

ROC AUC measures how well the model ranks positive examples above negative
examples.

Interpretation:

- `0.5`: random ranking
- `1.0`: perfect ranking
- higher is better

### Accuracy

Accuracy measures the fraction of final class predictions that are correct.

Accuracy is intuitive but can be misleading when class distribution is
imbalanced.

### Precision

Precision answers:

```text
When churn is predicted, how often is that prediction correct?
```

Higher precision means fewer false churn alerts.

### Recall

Recall answers:

```text
Out of all actual churn customers, how many were found?
```

Higher recall means fewer missed churn customers.

### F1 Score

F1 combines precision and recall into one score. It is useful when both false
positives and false negatives matter.

### Confusion Matrix

A confusion matrix shows:

- true negatives
- false positives
- false negatives
- true positives

It is useful for understanding the type of errors being made.

## Prediction Threshold

The model returns a probability. The threshold converts probability into the
final label.

Default threshold:

```python
prediction_threshold = 0.5
```

Tradeoff:

- lower threshold: higher recall, more false positives
- higher threshold: higher precision, more false negatives

Threshold selection should be based on business cost. If missing churn is more
expensive than a false alert, a lower threshold may be appropriate. If outreach
cost is high, a higher threshold may be better.

## Feature Defaults

The app can receive incomplete or invalid payloads. Training stores fallback
values for each feature:

- numeric feature default: median
- categorical feature default: mode

Median is less sensitive to outliers than mean. Mode is the most common
categorical value.

These defaults are saved inside the model artifact and used by
`FeaturePayloadBuilder`.

## Model Artifact

The trained artifact is stored at:

```text
models/model.joblib
```

The artifact is represented by `ModelArtifact` and persisted by
`ModelArtifactRepository`.

Artifact contents:

- artifact version
- fitted preprocessing/model pipeline
- feature column order
- numeric defaults
- categorical defaults
- target column metadata
- positive and negative label metadata
- prediction threshold

The project supports only the current unified artifact format. Older split
artifacts are intentionally unsupported.

### Artifact Contract

The artifact payload must contain a `pipeline` key.

```python
{
    "artifact_version": 2,
    "pipeline": fitted_pipeline,
    "feature_columns": [...],
    "numeric_defaults": {...},
    "categorical_defaults": {...},
    "target_column": "Churn",
    "positive_target_label": "Yes",
    "negative_target_label": "No",
    "prediction_threshold": 0.5,
}
```

Because the artifact is a serialized Python pipeline, compatible dependency
versions are important. Runtime dependencies are pinned in `requirements.txt`.

## Prediction Flow

Prediction is implemented in `src/predict.py`.

Flow:

1. `ChurnPredictor` loads `models/model.joblib`.
2. `ModelArtifactRepository` deserializes the artifact.
3. `FeaturePayloadBuilder` creates a one-row dataframe.
4. Payload columns are ordered according to the training feature list.
5. Missing or invalid values are replaced with saved defaults.
6. The saved pipeline runs `predict_proba`.
7. `ChurnPrediction` returns probability, numeric label, and text label.

Column order is part of the prediction contract. App-time input must be aligned
to the same feature structure used during training.

## Streamlit Application

The Streamlit app is implemented in `app/app.py` and launched through
`streamlit_app.py`.

Runtime responsibilities:

- load cleaned reference data
- build feature input widgets from the dataset schema
- call the local prediction service
- display prediction result
- display dataset exploration views

The Streamlit app does not train the model. It requires a saved artifact for
prediction.

If `models/model.joblib` is missing, run training before using prediction:

```bash
python -m src.train
```

## Training Workflow

Run training from the repository root:

```bash
python -m src.train
```

Training performs:

1. runtime directory creation
2. dataset loading
3. preprocessing
4. target encoding
5. stratified train/test split
6. pipeline creation
7. grid search with cross-validation
8. test-set evaluation
9. MLflow metric logging
10. artifact persistence to `models/model.joblib`

Current verified metrics after retraining:

```text
Best CV ROC AUC: 0.9597857798498137
Test ROC AUC:    0.9675864692986429
Test Accuracy:   0.9429429429429429
```

## Notebook Workflow

The notebook is located at:

```text
notebooks/train_churn_model.ipynb
```

The notebook mirrors the training workflow in smaller inspectable steps:

- data loading
- target inspection
- cleaning
- feature schema inference
- train/test split
- transformer creation
- pipeline creation
- grid search
- hyperparameter comparison
- test evaluation
- confusion matrix and classification metrics
- threshold comparison
- feature importance
- prediction defaults
- artifact saving
- artifact reload verification
- sample prediction
- optional SHAP explanation

## Dependencies

Runtime dependencies are pinned in:

```text
requirements.txt
```

Training and notebook dependencies are defined in:

```text
requirements-train.txt
```

`requirements-train.txt` includes runtime dependencies and adds:

- `mlflow`
- `openpyxl`
- `shap`
- `notebook`

Python 3.12 is used by the Docker runtime and is the recommended local runtime
for artifact compatibility.

## Local Setup

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install training dependencies:

```bash
pip install -r requirements-train.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Docker

Build and run the Streamlit service:

```bash
docker-compose up --build
```

Open:

```text
http://localhost:8501
```

The Docker image installs `requirements.txt` and runs the Streamlit entrypoint.
It is designed for serving the app, not for model experimentation.

## Deployment

The repository is shaped for Streamlit Community Cloud deployment.

Deployment settings:

- entrypoint: `streamlit_app.py`
- dependency file: `requirements.txt`
- required artifact: `models/model.joblib`
- required metrics report: `models/training_metrics.json`
- Python version: 3.12

Automated training:

- `.github/workflows/ci.yml` runs on every push to `main`
- the workflow installs `requirements-train.txt`
- it runs `python -m src.train`
- if training changes `models/model.joblib` or `models/training_metrics.json`,
  GitHub Actions commits the updated artifacts back to the branch
- pushes that only update trained artifacts are ignored to avoid a training loop

Streamlit Community Cloud setup:

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Select branch `main`.
5. Set the main file path to `streamlit_app.py`.
6. Open Advanced settings and choose Python 3.12.
7. Deploy the app.

If an existing Streamlit Cloud app is already running on a different Python
version, delete and redeploy the app with Python 3.12 selected in Advanced
settings. Streamlit Community Cloud does not use `runtime.txt` for Python
version selection.

After the app exists, Streamlit Cloud redeploys automatically when GitHub
receives new commits. Because the GitHub Actions workflow commits the retrained
`models/model.joblib` artifact after each source push, the deployed app receives
the latest trained model on the next Streamlit Cloud redeploy.

GitHub Pages is not suitable because this project requires Python execution.

## Runtime Simplification

The app does not use FastAPI, Celery, or Redis.

Those components are useful in larger distributed systems, but this project does
not require them because prediction is fast enough to run inside the Streamlit
process.

Benefits of the current runtime:

- one app process
- no internal HTTP dependency
- no task queue
- no worker service
- simpler free hosting path

## Operational Notes

- `models/model.joblib` must exist before app prediction is available.
- `mlruns/` stores local MLflow experiment data and is ignored by Git.
- `data/*.xlsx` is ignored by Git; `data/data.csv` is the expected dataset.
- The saved model artifact depends on compatible versions of sklearn, XGBoost,
  imbalanced-learn, numpy, pandas, and joblib.
- The current MLflow filesystem backend may emit a deprecation warning in newer
  MLflow versions. The warning is non-fatal for local use.
- For a production MLflow setup, use SQLite or a tracking server instead of the
  local filesystem backend.

## Project Contract

The current implementation assumes:

- Python 3.12 runtime
- dataset path: `data/data.csv`
- target column: `Churn`
- ID column: `customerID`
- positive label: `Yes`
- negative label: `No`
- model artifact path: `models/model.joblib`
- artifact format: unified payload with a `pipeline` key

Changing any of these assumptions requires updating `RuntimeConfig` and
retraining the model.
