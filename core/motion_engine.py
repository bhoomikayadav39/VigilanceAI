import cv2
import numpy as np
import sys
import time
import json
import os
from datetime import datetime

class MotionStateEngine:
    def __init__(self):
        self.project_name = "VigilanceAI - State Tracker Layer"
        self.motion_threshold = 12
        self.min_contour_area = 2000
        
        # Day 3 State Properties
        self.previous_position = None
        self.telemetry_log_path = "logs/motion_telemetry.json"
        
        # Ensure log directory exists
        import os
        os.makedirs("logs", exist_ok=True)

    def log_incident_event(self, area, velocity_delta):
        """Structures event metrics into a telemetry log for future LLM ingestion."""
        event_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": "KINETIC_ANOMALY",
            "metrics": {
                "spatial_area_pixels": int(area),
                "calculated_velocity_delta": float(velocity_delta)
            }
        }
        
        # Append to a local JSON tracking file
        try:
            logs = []
            if os.path.exists(self.telemetry_log_path):
                with open(self.telemetry_log_path, 'r') as f:
                    content = f.read()
                    if content:
                        logs = json.loads(content)
            
            logs.append(event_payload)
            # Keep only the last 20 events to save memory
            logs = logs[-20:]
            
            with open(self.telemetry_log_path, 'w') as f:
                json.dump(logs, f, indent=4)
            print(f"[LOGGED] Anomaly Saved -> Area: {area}px | Velocity Delta: {velocity_delta:.2f}")
        except Exception as e:
            print(f"[!] Log Failure: {e}")

    def run_engine(self):
        print(f"[*] Initializing {self.project_name}...")
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not camera.isOpened():
            camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not camera.isOpened():
            print("[!] FATAL: Camera feed blocked.")
            sys.exit(1)

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow("What Humans See (Motion Tracking)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("What AI Sees (Raw Velocity Delta)", cv2.WINDOW_NORMAL)

        success, frame_initial = camera.read()
        if not success:
            sys.exit(1)
            
        prev_frame = cv2.cvtColor(frame_initial, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)

        last_log_time = time.time()

        while True:
            success, frame = camera.read()
            if not success:
                break

            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

            frame_delta = cv2.absdiff(prev_frame, current_gray)
            _, threshold_mask = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)

            kernel = np.ones((30, 30), np.uint8)
            threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            largest_contour_area = 0
            current_center = None

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    if area > largest_contour_area:
                        largest_contour_area = area
                    
                    (x, y, w, h) = cv2.boundingRect(contour)
                    current_center = (x + w//2, y + h//2)
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "TRACKING STATE ACTIVE", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Day 3 Velocity Engine Math
            velocity_delta = 0.0
            if current_center is not None and self.previous_position is not None:
                # Euclidean distance calculation between frames: sqrt((x2-x1)^2 + (y2-y1)^2)
                velocity_delta = np.sqrt((current_center[0] - self.previous_position[0])**2 + 
                                         (current_center[1] - self.previous_position[1])**2)

            self.previous_position = current_center

            # Rate-limit logging to once every 1.5 seconds to avoid over-flooding the storage
            if largest_contour_area > 0 and (time.time() - last_log_time) > 1.5:
                if velocity_delta > 5.0: # Only log actual moving states
                    self.log_incident_event(largest_contour_area, velocity_delta)
                    last_log_time = time.time()

            cv2.imshow("What Humans See (Motion Tracking)", frame)
            cv2.imshow("What AI Sees (Raw Velocity Delta)", threshold_mask)

            prev_frame = current_gray

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = MotionStateEngine()
    engine.run_engine()