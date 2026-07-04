import cv2
import numpy as np
import sys

class MotionDeltaEngine:
    def __init__(self):
        self.project_name = "VigilanceAI - Motion Layer"
        self.motion_threshold = 12   # 🚀 Lowered to make internal motion turn white
        self.min_contour_area = 2000  # Track smaller movements easily

    def run_engine(self):
        print(f"[*] Initializing {self.project_name}...")
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not camera.isOpened():
            camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not camera.isOpened():
            print("[!] FATAL: Camera feed blocked.")
            sys.exit(1)

        # Force HD stream base
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 🚀 SETUP AUTO-STRETCH WINDOWS BEFORE THE LOOP
        cv2.namedWindow("What Humans See (Motion Tracking)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("What AI Sees (Raw Velocity Delta)", cv2.WINDOW_NORMAL)

        print("[✓] Motion Sensor Online. Stand completely still to calibrate background...")
        
        success, frame_initial = camera.read()
        if not success:
            sys.exit(1)
            
        prev_frame = cv2.cvtColor(frame_initial, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)

        while True:
            success, frame = camera.read()
            if not success:
                break

            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

            # Spatial Matrix Subtraction
            frame_delta = cv2.absdiff(prev_frame, current_gray)
            _, threshold_mask = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)

            # 🚀 Aggressive Morphological closing to fuse the broken outlines together
            kernel = np.ones((30, 30), np.uint8)
            threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > self.min_contour_area:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "MOTION VELOCITY DETECTED", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display with auto-stretch scaling
            cv2.imshow("What Humans See (Motion Tracking)", frame)
            cv2.imshow("What AI Sees (Raw Velocity Delta)", threshold_mask)

            prev_frame = current_gray

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = MotionDeltaEngine()
    engine.run_engine()