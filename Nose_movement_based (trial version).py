import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

class SmartNormalizedNoseTracker:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.smooth_vel_x = self.smooth_vel_y = 0.0
        self.prev_nose = None
        self.click_cooldown = 0
        
        # Movement bounds
        self.norm_factor = 0.018  # Tune: smaller = more sensitive
        
        # Landmarks
        self.NOSE_TIP = 1
        self.MOUTH_UPPER = 13
        self.MOUTH_LOWER = 14

    def detect_mouth_open(self, landmarks, h):
        return abs(landmarks[self.MOUTH_UPPER].y - landmarks[self.MOUTH_LOWER].y) * h > 25

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("👃 TINY nose movements = FULL screen control!")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = results.multi_face_landmarks[0].landmark
                nose = landmarks[self.NOSE_TIP]
                nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                cv2.circle(frame, (nose_x, nose_y), 8, (0, 255, 0), 2)
                
                # **SMART NORMALIZED MOVEMENT**
                if self.prev_nose:
                    # Raw pixel delta
                    dx_raw = nose_x - self.prev_nose[0]
                    dy_raw = nose_y - self.prev_nose[1]
                    
                    # Normalize to -1/+1 (small nose movement = full range)
                    dx_norm = np.tanh(dx_raw * self.norm_factor)  # Natural bounds
                    dy_norm = np.tanh(dy_raw * self.norm_factor)
                    
                    # Scale to screen movement
                    cursor_dx = dx_norm * screen_w * 0.25  # 25% screen per max tilt
                    cursor_dy = dy_norm * screen_h * 0.25
                    
                    # Responsive smoothing
                    self.smooth_vel_x = 0.75 * cursor_dx + 0.25 * self.smooth_vel_x
                    self.smooth_vel_y = 0.75 * cursor_dy + 0.25 * self.smooth_vel_y
                    
                    pyautogui.moveRel(int(self.smooth_vel_x), int(self.smooth_vel_y))
                
                self.prev_nose = (nose_x, nose_y)
                
                # Right click: mouth open
                if self.detect_mouth_open(landmarks, h) and time.time() - self.click_cooldown > 0.4:
                    pyautogui.rightClick()
                    self.click_cooldown = time.time()
                    cv2.putText(frame, "RIGHT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Movement indicator
                move_mag = np.sqrt(self.smooth_vel_x**2 + self.smooth_vel_y**2)
                cv2.putText(frame, f"Move: {move_mag:.0f}px", (50, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Smart Normalized Nose (TINY movements = FULL control)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SmartNormalizedNoseTracker()
    tracker.run()
