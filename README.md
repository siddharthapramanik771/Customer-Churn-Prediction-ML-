# Customer Churn Prediction App

This repository is now organized as a single-service Streamlit application that loads a trained churn model and runs predictions directly inside the app process. It is designed to be easy to run locally, easy to retrain, and easy to deploy on Streamlit Community Cloud from GitHub.

## What this project does

Given a customer record, the app predicts the probability of churn and returns:

- the churn probability
- a binary label
- the final class name: `Yes` or `No`

The repository still includes local training code, preprocessing logic, and model persistence, but the deployed app no longer depends on FastAPI, Celery, or Redis.

## Current architecture

The application now has two clear modes:

1. Training mode
   Uses the dataset, preprocessing pipeline, SMOTE, XGBoost, and MLflow logging.

2. App mode
   Loads the saved model bundle and performs direct in-process prediction inside Streamlit.

This is the structure:

```text
.
|-- app/
|   |-- app.py                 # Streamlit UI renderer and local prediction flow
|-- data/
|   |-- data.csv               # Dataset
|   |-- README.md
|-- models/
|   |-- model.joblib           # Saved trained model artifact
|-- src/
|   |-- config.py              # Central runtime configuration object
|   |-- model_bundle.py        # Saved model artifact structure
|   |-- predict.py             # Local predictor service
|   |-- preprocessing.py       # Data cleaning and feature preparation
|   |-- train.py               # Model training service
|-- .github/workflows/ci.yml   # Syntax-level CI check
|-- docker-compose.yml         # Single Streamlit service for local containers
|-- Dockerfile.streamlit
|-- requirements.txt           # Lean runtime dependencies for the deployed app
|-- requirements-train.txt     # Extra dependencies needed for retraining
`-- streamlit_app.py           # Root Streamlit entrypoint for deployment
```

## Object-oriented structure

The project was refactored to use small service classes instead of scattered procedural logic.

### `src/config.py`

Contains `RuntimeConfig`, which owns:

- project paths
- data path
- model path
- MLflow directory
- target labels
- prediction threshold

`RUNTIME_CONFIG` is the shared application configuration object used across the repo.

### `src/preprocessing.py`

Contains:

- `DataPreprocessor`
- `FeatureSchema`
- `FeatureDefaults`

Responsibilities:

- normalize string/object columns
- convert numeric-looking text to real numeric types
- remove the ID column
- drop missing rows
- infer numeric and categorical feature groups
- encode the target column
- derive default fallback values for prediction-time payload filling

### `src/model_bundle.py`

Contains `ModelArtifact`, which defines the saved model structure.

It stores:

- the fitted pipeline
- feature column order
- numeric defaults
- categorical defaults

It also supports the legacy saved format in case older model bundles still exist.

### `src/train.py`

Contains:

- `TrainingMetrics`
- `ChurnModelTrainer`

Responsibilities:

- load and clean data
- split train/test sets
- build the full modeling pipeline
- run `GridSearchCV`
- log metrics to MLflow
- save the final model artifact

### `src/predict.py`

Contains:

- `FeaturePayloadBuilder`
- `LegacyPredictorAdapter`
- `ChurnPredictor`

Responsibilities:

- load the saved artifact lazily
- align input payloads to training schema
- fill missing fields from saved defaults
- run probability prediction

### `app/app.py`

Contains:

- `PredictionResult`
- `ReferenceDataService`
- `LocalPredictionService`
- `DashboardRenderer`

Responsibilities:

- load dataset preview data
- render the feature input form from the real schema
- run prediction directly in process
- display model output and data exploration views

## Dataset expectations

Default dataset path:

`data/data.csv`

Current target configuration:

- Target column: `Churn`
- Positive label: `Yes`
- Negative label: `No`
- ID column removed before training: `customerID`

The current dataset contains 33 raw columns and is cleaned down to the feature set used for modeling.

## Preprocessing behavior

The preprocessing layer does a few important things:

1. Strips whitespace from column names.
2. Trims string values.
3. Converts blank strings to missing values.
4. Detects numeric-looking object columns and converts them.
5. Drops missing rows.
6. Removes the ID column.
7. Splits features into numeric and categorical groups automatically.

This matters because fields like `TotalRevenue` may arrive as text in the raw CSV and should be treated as numeric features.

## Training pipeline

Training still happens locally through `src/train.py`.

The model pipeline is:

`ColumnTransformer -> SMOTE -> XGBClassifier`

Hyperparameter search currently covers:

- `n_estimators`: `100`, `200`
- `max_depth`: `4`, `6`
- `learning_rate`: `0.01`, `0.05`

Training logs metrics to MLflow and saves the final artifact to:

`models/model.joblib`

## Verified training result

The current training flow was re-run successfully after the refactor and produced:

- Best CV ROC AUC: `0.9598`
- Test ROC AUC: `0.9676`
- Test Accuracy: `0.9429`

## App behavior

The deployed Streamlit app no longer sends requests to a backend service.

Instead, it:

1. loads `models/model.joblib`
2. builds a form from the cleaned dataset schema
3. predicts directly using `ChurnPredictor`
4. shows:
   - predicted class
   - churn probability
   - latency
   - raw JSON output
5. keeps the data explorer for preview, summary stats, distributions, and correlation view

That makes deployment much easier because there is only one process to host.

## Local setup

### Run the app locally

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Start the app from the repository root:

```bash
streamlit run streamlit_app.py
```

### Retrain the model locally

Install training dependencies:

```bash
pip install -r requirements-train.txt
```

Run training:

```bash
python -m src.train
```

## Docker

This repo now includes a single-service Docker setup for the Streamlit app.

Start it with:

```bash
docker-compose up --build
```

Then open:

`http://localhost:8501`

## Streamlit Community Cloud deployment

This repository is now shaped specifically to deploy cleanly on Streamlit Community Cloud.

Important deployment files:

- entrypoint: `streamlit_app.py`
- dependency file: `requirements.txt`
- model artifact: `models/model.joblib`

### Deploy steps

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Click `Create app`.
4. Select your GitHub repository.
5. Set the entrypoint file to `streamlit_app.py`.
6. Deploy.

That is enough for this version of the project because prediction now happens directly inside Streamlit.

## Why this deploys better than the old version

The previous architecture needed:

- a FastAPI service
- a Celery worker
- Redis
- task polling between UI and backend

That stack is workable locally, but it is more fragile on free hosting.

The current version avoids all of that at runtime:

- no internal network calls
- no queue
- no separate worker process
- no Redis dependency

This is much more suitable for Streamlit Community Cloud.

## Dependency split

There are now two dependency files on purpose.

### `requirements.txt`

Used for the deployed Streamlit app and lean local runtime.

### `requirements-train.txt`

Used when you want to retrain the model locally. It extends runtime dependencies and adds:

- `mlflow`
- `openpyxl`
- `shap`

This keeps Streamlit deployment lighter while preserving your training workflow.

## CI

The GitHub Actions workflow still performs a syntax-level compile check over tracked Python files.

## Free hosting note

GitHub is a great place to store the code, but GitHub Pages is not suitable for this app because GitHub Pages only hosts static sites.

For this repo, the best free hosting option is Streamlit Community Cloud because it is built for Python Streamlit apps and deploys directly from GitHub.

## What still requires your account

I can refactor and prepare the repo, but I cannot complete the final Streamlit Community Cloud deployment from here unless you provide access through your own GitHub and Streamlit accounts.

The codebase is now prepared for that final step.
