from app.celery_app import celery_app
from app.utils import download_video, save_result
from app.detection import VideoProcessor
from pathlib import Path

processor = VideoProcessor()

@celery_app.task(name="process_video")
def process_video(video_url: str):
    filepath = download_video(video_url)
    processed_video_path, metadata_path = processor.run_detection(filepath)
    result_video_url, result_metadata_url = save_result(processed_video_path, metadata_path)
    return {
        "video_id": Path(filepath).stem,
        "video_url": result_video_url,
        "metadata_url": result_metadata_url
    }