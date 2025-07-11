from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, HttpUrl
from app.celery_app import celery_app
from app.tasks import process_video
from celery.result import AsyncResult
from pathlib import Path
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FastAPI()
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./processed"))
host = os.getenv("HOST")
port = os.getenv("PORT")

# Pydantic will verify that the input is a valid url
class VideoInput(BaseModel):
    video_url: HttpUrl


@app.post("/submit")
def submit_video(input: VideoInput):
    task = process_video.delay(str(input.video_url))
    return {"task_id": task.id}


@app.get("/status/{task_id}")
def get_task_status(task_id: str):
    """
    Returns the status of the task given the task_id.
    1. Pending: Returns processing status
    2. Failed: Returns failed status along with the error that occurred
    3. Success: Returns success status along with video_id(can be used later in the /processed/{video_id} endpoint),
       video_url and metadata_url which can be used to immediately access the processed video and metadata
    """
    res = AsyncResult(task_id, app=celery_app)
    if res.state == "PENDING":
        return {"status": "processing"}
    elif res.state == "FAILURE":
        return {"status": "failed", "error": str(res.result)}
    elif res.state == "SUCCESS":
        return {
            "status": "success",
            "video_id": res.result["video_id"],
            "video_url": res.result["video_url"],
            "metadata_url": res.result["metadata_url"],
        }

@app.get("/processed/{video_id}")
def get_processed(video_id: str):
    return {
        "video_url": f"http://{host}:{port}/stream/{video_id}",
        "download_video_url": f"http://{host}:{port}/download/{video_id}",
        "metadata_url": f"http://{host}:{port}/metadata/{video_id}",
    }

@app.get("/download/{video_id}")
def download_processed(video_id: str):
    filepath = PROCESSED_DIR / f"processed_{video_id}.mp4"
    
    if not filepath.exists():
        raise HTTPException(404, detail="file not found")
    return FileResponse(
        path=filepath,
        media_type="application/octet-stream",
        filename=filepath.name,
        headers={"Content-Disposition": f'attachment; filename="{filepath.name}"'}
    )

@app.get("/download/{video_id}/metadata")
def download_metadata(video_id: str):
    filepath = PROCESSED_DIR / f"processed_{video_id}.json"
    if not filepath.exists():
        raise HTTPException(404, "metadata not found")
    return FileResponse(
        path=filepath,
        media_type="application/json",
        filename=filepath.name,
        headers={"Content-Disposition": f'attachment; filename="{filepath.name}"'}
    )

@app.get("/stream/{video_id}")
def stream_video(video_id: str):
    filepath = PROCESSED_DIR / f"processed_{video_id}.mp4"
    if not filepath.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(
        path=filepath,
        media_type="video/mp4",
        filename=filepath.name,
        headers={"Content-Disposition": "inline"}
    )

@app.get("/metadata/{video_id}")
def get_metadata(video_id: str):
    filepath = PROCESSED_DIR / f"processed_{video_id}.json"
    if not filepath.exists():
        raise HTTPException(404, "file not found")
    with open(filepath, "r") as f:
        data = json.load(f)

    return JSONResponse(content=data)