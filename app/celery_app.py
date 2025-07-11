from celery import Celery
from dotenv import load_dotenv
import os

load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_BACKEND_URL = os.getenv("CELERY_BACKEND_URL", "redis://localhost:6379/1")

celery_app = Celery("gaze_tasks", broker=CELERY_BROKER_URL, backend=CELERY_BACKEND_URL)
celery_app.autodiscover_tasks(["app"])
