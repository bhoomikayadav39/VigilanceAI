import cv2
import numpy as np
import sys

class LiveVisionEngine:
    def __init__(self):
        self.project_name = "VigilanceAI"
        self.danger_limit_count = 3  
        # We start with a 0-179 range so it catches everything, then we invert it!
        self.hue_min, self.hue_max = 0, 179      
        self.sat_min, self.val_min = 20, 20

    def run_sensor(self):
        print(f"[*] Booting {self.project_name} Ultra-Clear Pipeline...")
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not camera.isOpened():
            camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        if not camera.isOpened():
            print("\n[!] FATAL: Webcam inaccessible.")
            sys.exit(1)
        
        cv2.namedWindow("AI Control Panel", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Min Hue", "AI Control Panel", self.hue_min, 179, lambda x: None)
        cv2.createTrackbar("Max Hue", "AI Control Panel", self.hue_max, 179, lambda x: None)

        print("[✓] Camera online. Noise suppression layer deployed.")

        while True:
            success, frame = camera.read()
            if not success:
                break

            h_min = cv2.getTrackbarPos("Min Hue", "AI Control Panel")
            h_max = cv2.getTrackbarPos("Max Hue", "AI Control Panel")

            # NEW STEP 1: Gaussian Blur (Smudges out the sharp digital static noise completely)
            clean_frame = cv2.GaussianBlur(frame, (15, 15), 0)

            # Convert blurred frame to HSV matrix mapping
            hsv_matrix = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2HSV)
            
            lower_bound = np.array([h_min, self.sat_min, self.val_min])
            upper_bound = np.array([h_max, 255, 255])
            color_mask = cv2.inRange(hsv_matrix, lower_bound, upper_bound)

            # NEW STEP 2: Bitwise NOT Operation (Flips the colors!)
            # This turns the black silhouette of your body into crisp, clean WHITE 
            # and drops the white background noise down to pitch BLACK.
            color_mask = cv2.bitwise_not(color_mask)

            # NEW STEP 3: Morphological Closing (Fills in any remaining holes inside your shape)
            kernel = np.ones((40, 40), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tracked_objects_count = 0

            for contour in contours:
                if cv2.contourArea(contour) > 5000:  # Increased size threshold to only track BIG objects (You!)
                    tracked_objects_count += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "TARGET DETECTED", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if tracked_objects_count > self.danger_limit_count:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                cv2.putText(frame, "CRISIS OVERLOAD", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.imshow("What Humans See (Real-time Detection)", frame)
            cv2.imshow("What AI Sees (Pure Isolated Binary Color)", color_mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = LiveVisionEngine()
    engine.run_sensor()