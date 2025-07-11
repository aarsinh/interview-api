# interview-api

A FastAPI-based backend for asynchronous video processing, featuring Celery task management, video and metadata storage, and RESTful endpoints for status tracking, streaming, and downloads.

## Requirements

- Python 3.13+
- Redis (for Celery broker and backend)
- See `pyproject.toml` for Python dependencies

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd interview-api
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or, if using PEP 621/pyproject.toml:
   ```bash
   pip install .
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   HOST=localhost
   PORT=8000
   PROCESSED_DIR=./processed
   ```

5. **Start Redis server** (if not already running):
   ```bash
   redis-server
   ```

## Running the Application

1. **Start the Celery worker:**
   ```bash
   celery -A app.celery_app.celery_app worker --loglevel=info
   ```

2. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### 1. Submit a Video for Processing

**POST** `/submit`

- **Body:**  
  ```json
  { "video_url": "https://example.com/video.mp4" }
  ```
- **Response:**  
  ```json
  { "task_id": "<task_id>" }
  ```

### 2. Check Task Status

**GET** `/status/{task_id}`

- **Response:**  
  - `{"status": "processing"}`
  - `{"status": "failed", "error": "<error>"}`  
  - `{"status": "success", "video_id": "<id>", "video_url": "...", "metadata_url": "..."}`

### 3. Get Processed Video and Metadata URLs

**GET** `/processed/{video_id}`

- **Response:**  
  ```json
  {
    "video_url": "http://<host>:<port>/stream/<video_id>",
    "download_video_url": "http://<host>:<port>/download/<video_id>",
    "metadata_url": "http://<host>:<port>/metadata/<video_id>"
  }
  ```

### 4. Stream Processed Video

**GET** `/stream/{video_id}`  
Streams the processed video in-browser.

### 5. Download Processed Video

**GET** `/download/{video_id}`  
Downloads the processed video file.

### 6. Download Metadata

**GET** `/download/{video_id}/metadata`  
Downloads the metadata JSON file.

### 7. Get Metadata as JSON

**GET** `/metadata/{video_id}`  
Returns the metadata as a JSON response.

## Directory Structure

```
interview-api/
  app/
  core/
  processed/           # Processed videos and metadata
  main.py              # FastAPI app entrypoint
  pyproject.toml
  README.md
  ...
```
