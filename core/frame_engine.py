import cv2
import numpy as np

def stream_and_track_motion():
    # 1. Boot up local hardware camera (0 is default webcam)
    camera = cv2.VideoCapture(0)
    
    print("[*] Sensor Active. Wave your hand to see the pixel math...")
    
    # Read the very first baseline frame to use as memory
    success, baseline_frame = camera.read()
    if not success:
        print("[!] Camera matrix sensor failed.")
        return
        
    # Convert to grayscale to focus on light changes, not heavy colors
    prev_gray = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        success, current_frame = camera.read()
        if not success:
            break
            
        # Convert the active frame to grayscale matrix
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # CORE MATH CONCEPT: Subtract matrix A from matrix B
        # If pixels are identical, result = 0 (black). If they changed, result > 0 (white).
        frame_difference = cv2.absdiff(prev_gray, current_gray)
        
        # Clean the noise: Any pixel shift over 25 gets turned into solid white (255)
        _, clean_threshold = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate total motion volume percentage
        motion_pixel_count = np.sum(clean_threshold == 255)
        total_pixels = clean_threshold.size
        motion_percentage = (motion_pixel_count / total_pixels) * 100
        
        # If movement is real (more than 1% of screen), highlight it
        if motion_percentage > 1.0:
            print(f"[!] MOTION DETECTED: Pixel Matrix shifted by {motion_percentage:.2f}%")
            # Draw a visual neon border around the active window to alert user
            cv2.putText(current_frame, "ANOMALY: RETAINING STREAM", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # Show your followers the visual windows
        cv2.imshow("What Humans See (Camera feed)", current_frame)
        cv2.imshow("What AI Sees (Pure Matrix Math Difference)", clean_threshold)
        
        # Update memory for the next frame subtraction pass
        prev_gray = current_gray
        
        # Press 'q' to safely kill the system loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_and_track_motion()