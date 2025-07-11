import cv2
from core.detect import CheatingDetector, detect_gazes
from core.gaze_calibrator import GazeCalibrator
from pathlib import Path
from datetime import datetime
import json
from core.utils import draw_gaze_with_alerts
import numpy as np
import time


class VideoProcessor:
    def __init__(self):
        self.cheating_detector = CheatingDetector()
        self.gaze_calibrator = GazeCalibrator()
        
    def run_detection(self, input_path: str) -> tuple[str, str]:
        '''
        Run gaze detection on the downloaded video. This does two things:
        1. Draws overlays on the video
        2. Stores the result in a json file
        Both results are initially stored in tmp directory, later shifted to a mounted volume
        '''
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            RuntimeError(f"Failed to open file: {input_path}")

        print("video opened" ,flush=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        print(f"Total frames: {total_frames}, FPS: {fps}", flush=True)
        print("Starting frame loop...", flush=True)
        
        analysis_data = {
            "total_frames": total_frames,
            "fps": fps,
            "duration": total_frames / fps,
            "cheating_events": [],
            "frame_analysis": [],
            "final_summary": [],
            "start_time": datetime.now()
        }
        
        input_id = Path(input_path).stem
        processed_video_path = Path("/tmp") / f"processed_{input_id}.mp4"
        metadata_path = Path("/tmp") / f"{input_id}_metadata.json"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (int(width), int(height))
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, frame_size)
        
        frame_count = 0
        last_logged = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps

                processed_frame, frame_analysis = self._process_single_frame(frame, timestamp, frame_count)
                
                if frame_count % 30 == 0 and frame_analysis:
                    analysis_data["frame_analysis"].append(frame_analysis)
                    if frame_analysis.get('alert_level') in ['CHEATING_DETECTED', 'SUSPICIOUS_BEHAVIOR']:
                        cheating_event = {
                            "timestamp": timestamp,
                            "frame_number": frame_count,
                            "alert_level": frame_analysis["alert_level"],
                            "suspicion_score": frame_analysis.get("suspicion_score", 0)
                        }
                        
                        analysis_data["cheating_events"].append(cheating_event)
                
                out.write(processed_frame)
                frame_count += 1

                if time.time() - last_logged >= 10:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", flush=True)
                    last_logged = time.time()
                    
        except Exception as e:
            print(f"Exception occurred: {e}")
            
        finally:
            cap.release()
            out.release()
            
        if self.cheating_detector:
            final_summary = self.cheating_detector.get_session_summary()
            analysis_data["final_summary"] = final_summary
            
        with open(metadata_path, "w") as f:
            json.dump(analysis_data, f, default=lambda x: x.isoformat() if isinstance(x, datetime) else str(x))
        return str(processed_video_path), str(metadata_path)
    
    def _process_single_frame(self, frame, timestamp, frame_number):
        frame_analysis = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'alert_level': 'NO_ALERT',
            'suspicion_score': 0
        }
        
        try:
            gazes = detect_gazes(frame)
            for gaze in gazes:
                if self.cheating_detector and self.gaze_calibrator:
                    facial_structure = self.gaze_calibrator.analyze_facial_structure(gaze["face"])
                    if facial_structure:
                        self.gaze_calibrator.update_baseline(gaze, facial_structure)
                        calibrated_gaze = self.gaze_calibrator.calibrated_gaze_prediction(gaze, facial_structure)
                        
                        if calibrated_gaze and self.gaze_calibrator.is_calibrated():
                            detection_result = self.cheating_detector.analyze_gaze_behavior(
                                calibrated_gaze, facial_structure, timestamp
                            )
                            
                            frame_analysis.update({
                                'alert_level': detection_result.get('alert_level', 'NO_ALERT'),
                                'suspicion_score': detection_result.get('suspicion_score', 0),
                                'is_looking_away': detection_result.get('is_looking_away', False),
                                'details': {
                                    'yaw': calibrated_gaze.get('calibrated_yaw', 0),
                                    'pitch': calibrated_gaze.get('calibrated_pitch', 0),
                                    'confidence': calibrated_gaze.get('confidence', 0)
                                }
                            })
                            
                            frame = draw_gaze_with_alerts(frame, gaze, detection_result)
                        else:
                            frame = self._draw_simple_gaze(frame, gaze)
                    else:
                        frame = self._draw_simple_gaze(frame, gaze)
                else:
                    frame = self._draw_simple_gaze(frame, gaze)
                    
                timestamp_text = f"Time: {timestamp:.2f}s | Frame: {frame_number}"
                cv2.putText(frame, timestamp_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            cv2.putText(frame, f"Processing Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return frame, frame_analysis
    
    def _draw_simple_gaze(self, img, gaze):
        """Simple gaze visualization for fallback"""
        face = gaze["face"]
        x_min = int(face["x"] - face["width"] / 2)
        x_max = int(face["x"] + face["width"] / 2)
        y_min = int(face["y"] - face["height"] / 2)
        y_max = int(face["y"] + face["height"] / 2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        arrow_length = 100
        dx = -arrow_length * np.sin(gaze["yaw"]) * np.cos(gaze["pitch"])
        dy = -arrow_length * np.sin(gaze["pitch"])
        
        center_x, center_y = int(face["x"]), int(face["y"])
        end_x, end_y = int(center_x + dx), int(center_y + dy)
        
        cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), 
                       (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.3)
        return img