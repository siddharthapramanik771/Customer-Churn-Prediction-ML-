from fastapi import FastAPI
from pydantic import BaseModel
from celery.result import AsyncResult
from tasks.worker import celery_app
from tasks.tasks import predict_task

app = FastAPI()

class Payload(BaseModel):
    # flexible payload; keys should match training columns
    data: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_async")
def predict_async(p: Payload):
    task = predict_task.delay(p.data)
    return {"task_id": task.id}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if res.ready():
        return {"status": "done", "result": res.result}
    return {"status": "processing"}
