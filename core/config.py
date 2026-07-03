import cv2
import numpy as np
import sys

class LiveVisionEngine:
    def __init__(self):
        self.project_name = "VigilanceAI"
        self.danger_limit_count = 3  
        self.hue_min, self.hue_max = 35, 85      
        self.sat_min, self.val_min = 40, 40

    def run_sensor(self):
        print(f"[*] Booting {self.project_name} Camera Pipeline...")
        
        # Forces native Windows DirectShow backend to ensure the camera triggers cleanly
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not camera.isOpened():
            print("[*] Trying secondary channel index...")
            camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        if not camera.isOpened():
            print("\n[!] FATAL: Could not access your webcam stream.")
            print("[!] Check: Is your physical Lenovo slider privacy shutter slid shut over the lens?")
            sys.exit(1)
        
        # Setup visual control bars
        cv2.namedWindow("AI Control Panel", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Min Hue", "AI Control Panel", self.hue_min, 179, lambda x: None)
        cv2.createTrackbar("Max Hue", "AI Control Panel", self.hue_max, 179, lambda x: None)

        print("[✓] Camera stream online. Wave at the camera lens!")

        while True:
            success, frame = camera.read()
            if not success:
                print("[!] Error reading active frame matrix stream.")
                break

            h_min = cv2.getTrackbarPos("Min Hue", "AI Control Panel")
            h_max = cv2.getTrackbarPos("Max Hue", "AI Control Panel")

            # Convert to HSV matrix mapping
            hsv_matrix = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            lower_bound = np.array([h_min, self.sat_min, self.val_min])
            upper_bound = np.array([h_max, 255, 255])
            color_mask = cv2.inRange(hsv_matrix, lower_bound, upper_bound)

            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tracked_objects_count = 0

            for contour in contours:
                if cv2.contourArea(contour) > 500:  
                    tracked_objects_count += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Obj #{tracked_objects_count}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if tracked_objects_count > self.danger_limit_count:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                cv2.putText(frame, "SYSTEM OVERLOAD: CRISIS ALERT", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.imshow("What Humans See (Real-time Detection)", frame)
            cv2.imshow("What AI Sees (Pure Isolated Binary Color)", color_mask)

            # Press 'q' to close the app windows safely
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = LiveVisionEngine()
    engine.run_sensor()