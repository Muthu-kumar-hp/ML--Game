import cv2
import os
import numpy as np
import time

class DataCollector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.gestures = ['stone', 'paper', 'scissors']
        self.data_dir = 'dataset'   # ✅ match training script
        self.img_size = (64, 64)
        
        # Create directories
        for gesture in self.gestures:
            os.makedirs(f'{self.data_dir}/{gesture}', exist_ok=True)
    
    def collect_data(self, gesture_name, num_samples=500):
        """Collect training data for a specific gesture"""
        print(f"Collecting data for: {gesture_name}")
        print("Press SPACE to start collecting, ESC to stop")
        
        sample_count = 0
        collecting = False
        
        while sample_count < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)  # mirror effect
            
            # Hand ROI (adjust as needed)
            x, y, w, h = 100, 100, 300, 300
            hand_region = frame[y:y+h, x:x+w]
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Show status
            cv2.putText(frame, f'Gesture: {gesture_name}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'Samples: {sample_count}/{num_samples}', (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if collecting:
                cv2.putText(frame, 'COLLECTING...', (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Preprocess
                gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, self.img_size)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Save
                filename = f'{self.data_dir}/{gesture_name}/img_{sample_count:04d}.jpg'
                success = cv2.imwrite(filename, gray)
                if success:
                    sample_count += 1
                else:
                    print(f"⚠️ Failed to save {filename}")
                
                time.sleep(0.1)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                collecting = not collecting
                print(f"Collecting: {collecting}")
            elif key == 27:  # ESC
                break
        
        print(f"✅ Collected {sample_count} samples for {gesture_name}")
    
    def collect_all_gestures(self, samples_per_gesture=500):
        try:
            for gesture in self.gestures:
                input(f"\nPress Enter to start collecting data for '{gesture}'...")
                self.collect_data(gesture, samples_per_gesture)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("🎉 Data collection complete!")

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_all_gestures(samples_per_gesture=300)

