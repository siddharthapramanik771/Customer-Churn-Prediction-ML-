# Customer Churn Prediction System

## 🚀 Overview
End-to-end ML system with:
- Feature pipeline (ColumnTransformer)
- Imbalance handling (SMOTE)
- XGBoost + GridSearchCV
- Experiment tracking & registry (MLflow)
- Explainability (SHAP)
- Real-time API (FastAPI)
- Async inference (Celery + Redis)
- Dashboard (Streamlit)
- Containerization (Docker + docker-compose)
- CI/CD (GitHub Actions)

## 🧱 Architecture
Data → Feature Pipeline → Training → MLflow (tracking/registry)
→ API (FastAPI) → Queue (Redis) → Worker (Celery)
→ Predictions → Dashboard (Streamlit)

## ⚙️ Quick Start
1) Place dataset as `data/data.csv` (current dataset target: `Churn`)
2) Update [src/config.py](</c:/Users/SIDDHARTA PRAMANIK/PycharmProjects/Customer Churn Prediction (ML)/src/config.py>) if your dataset path, target column, or model settings change.
3) Build & run:
   docker-compose up --build

## 🔌 Endpoints
- GET  /health
- POST /predict_async  {json features}
- GET  /result/{task_id}

## 🖥️ Apps
- API: http://localhost:8000
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000
