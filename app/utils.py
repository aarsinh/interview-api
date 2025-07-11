import requests
import uuid
from pathlib import Path
import shutil
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", "/tmp/downloaded"))
DOWNLOAD_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./processed"))
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def download_video(video_url: str) -> str:
    '''
    download video from url using requests lib and store it in tmp folder
    '''
    try:
        video_id = str(uuid.uuid4())
        filename = f"{video_id}.mp4"
        filepath = DOWNLOAD_DIR / filename
        
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        return str(filepath)
    
    except Exception as e:
        raise RuntimeError(f"failed to download video from url: {e}")

def save_result(video_path: str, metadata_path: str) -> tuple[str, str]:
    '''
    save the result to PROCESSED_DIR with video id and return the url for accessing the files
    '''
    try:
        video_id = Path(video_path).stem
        processed_video_path = PROCESSED_DIR / f'{video_id}.mp4'
        processed_metadata_path = PROCESSED_DIR / f"{video_id}.json"
        shutil.move(video_path, processed_video_path)
        shutil.move(metadata_path, processed_metadata_path)
        
        video_url = f"/processed/{processed_video_path.name}"
        metadata_url = f"/processed/{processed_metadata_path.name}"

        return video_url, metadata_url
    except Exception as e:
        raise RuntimeError(f"failed to save results: {e}")