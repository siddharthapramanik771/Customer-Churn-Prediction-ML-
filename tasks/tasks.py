from .worker import celery_app
from src.config import PREDICTION_THRESHOLD
from src.predict import predict_proba_single

@celery_app.task(name="predict_task")
def predict_task(data: dict):
    proba = predict_proba_single(data)
    return {"churn_probability": proba, "label": int(proba > PREDICTION_THRESHOLD)}
