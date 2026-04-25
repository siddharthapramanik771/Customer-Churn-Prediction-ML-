from celery import Celery
from src.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

# Ensure the task modules are imported by the worker so tasks get registered.
celery_app = Celery(
    "churn_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["tasks.tasks"],
)

# also import tasks module when this file is imported (safe because
# `tasks.tasks` imports `celery_app` from here)
try:
    import tasks.tasks  # noqa: F401
except Exception:
    pass
