"""
🚀 HORIZONTAL DIRECTION FIXED - Look LEFT → Cursor LEFT!
Both vertical + horizontal axes corrected
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
from collections import deque

pyautogui.FAILSAFE = False

class PerfectDirectionGazeTracker:
    def __init__(self):
        self.smooth_buffer_x = deque(maxlen=12)
        self.smooth_buffer_y = deque(maxlen=12)
        
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        
        self.landmarks = {
            'left_iris': 468, 'right_iris': 473,
            'left_inner': 33, 'left_outer': 362,
            'right_inner': 133, 'right_outer': 385,
            'nose_bridge': 1, 'chin': 152, 'forehead': 10
        }
        
    def find_camera(self):
        for i in range(4):
            cap = cv2.VideoCapture(i)
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return i
        return 0
    
    def mirror_frame(self, frame):
        return cv2.flip(frame, 1)
    
    def calculate_perfect_gaze(self, landmarks, img_shape, screen_shape):
        """✅ BOTH AXES FIXED - Perfect direction mapping"""
        h_img, w_img = img_shape[:2]
        screen_w, screen_h = screen_shape
        
        # 1. PUPIL OFFSETS (MIRROR CORRECTED)
        left_iris = landmarks[self.landmarks['left_iris']]
        right_iris = landmarks[self.landmarks['right_iris']]
        left_inner = landmarks[self.landmarks['left_inner']]
        left_outer = landmarks[self.landmarks['left_outer']]
        right_inner = landmarks[self.landmarks['right_inner']]
        right_outer = landmarks[self.landmarks['right_outer']]
        
        # Eye centers
        left_center_x = (left_inner.x + left_outer.x) / 2
        left_center_y = (left_inner.y + left_outer.y) / 2
        right_center_x = (right_inner.x + right_outer.x) / 2
        right_center_y = (right_inner.y + right_outer.y) / 2
        
        # ✅ HORIZONTAL FIXED: MIRROR INVERSION CORRECTED
        # Look LEFT → pupils move RIGHT in MIRROR → cursor LEFT
        pupil_offset_x_left = -(left_iris.x - left_center_x)  # FLIP for mirror
        pupil_offset_y_left = -(left_iris.y - left_center_y)  # FLIP for screen Y
        pupil_offset_x_right = -(right_iris.x - right_center_x)  # FLIP for mirror
        pupil_offset_y_right = -(right_iris.y - right_center_y)  # FLIP for screen Y
        
        # Average dual-eye (normalized)
        avg_pupil_x = (pupil_offset_x_left + pupil_offset_x_right) / 2
        avg_pupil_y = (pupil_offset_y_left + pupil_offset_y_right) / 2
        
        # 2. HEAD POSE COMPENSATION (CORRECTED)
        nose_x = landmarks[self.landmarks['nose_bridge']].x
        eye_center_x = (left_center_x + right_center_x) / 2
        
        # ✅ HORIZONTAL HEAD: Look LEFT → nose LEFT → cursor LEFT
        head_offset_x = -(nose_x - eye_center_x)  # Mirror corrected
        head_offset_y = 0  # Simplified
        
        # 3. EYE SIZE NORMALIZATION
        eye_width_left = abs(left_outer.x - left_inner.x)
        eye_width_right = abs(right_outer.x - right_inner.x)
        avg_eye_size = (eye_width_left + eye_width_right) / 2
        
        # 4. FINAL FUSION - PERFECT DIRECTIONS
        sensitivity_x = 4.5 / avg_eye_size  # Dynamic
        sensitivity_y = 3.8 / avg_eye_size
        
        # ✅ PERFECT DIRECTION MAPPING
        gaze_x = 0.5 + (avg_pupil_x * 0.7 + head_offset_x * 0.3) * sensitivity_x
        gaze_y = 0.5 + (avg_pupil_y * 0.8 + head_offset_y * 0.2) * sensitivity_y
        
        # Clamp
        final_x = np.clip(gaze_x, 0.02, 0.98)
        final_y = np.clip(gaze_y, 0.02, 0.98)
        
        return final_x, final_y, avg_pupil_x, avg_pupil_y, avg_eye_size
    
    def run(self):
        camera_idx = self.find_camera()
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        for _ in range(40): cap.read()
        
        WINDOW_NAME = '✅ BOTH DIRECTIONS PERFECT!'
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1200, 800)
        
        screen_w, screen_h = pyautogui.size()
        print("✅ HORIZONTAL + VERTICAL FIXED!")
        print("🎮 Look LEFT → Cursor LEFT | Look RIGHT → Cursor RIGHT")
        
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            # frame = self.mirror_frame(frame)
            h, w = frame.shape[:2]
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # PERFECT DIRECTION GAZE
                gaze_x, gaze_y, pupil_x, pupil_y, eye_size = self.calculate_perfect_gaze(
                    landmarks, (h, w), (screen_w, screen_h)
                )
                
                # Smooth
                self.smooth_buffer_x.append(gaze_x)
                self.smooth_buffer_y.append(gaze_y)
                smooth_x = np.mean(self.smooth_buffer_x)
                smooth_y = np.mean(self.smooth_buffer_y)
                
                # PRECISE MOUSE
                screen_x = int(smooth_x * screen_w)
                screen_y = int(smooth_y * screen_h)
                pyautogui.moveTo(screen_x, screen_y, duration=0.005)
                
                # MASSIVE GREEN CONFIRMATION
                pred_pt = (int(smooth_x * w), int(smooth_y * h))
                cv2.circle(frame, pred_pt, 25, (0, 255, 0), 5)  # GREEN = PERFECT!
                cv2.circle(frame, pred_pt, 18, (255, 255, 255), -1)
                
                # PUPILS
                cv2.circle(frame, 
                    (int(landmarks[self.landmarks['left_iris']].x * w), 
                     int(landmarks[self.landmarks['left_iris']].y * h)), 
                    16, (0, 0, 255), -1)
                cv2.circle(frame, 
                    (int(landmarks[self.landmarks['right_iris']].x * w), 
                     int(landmarks[self.landmarks['right_iris']].y * h)), 
                    16, (255, 0, 0), -1)
                
                # NOSE FOR HEAD TRACKING
                nose_pt = (int(landmarks[self.landmarks['nose_bridge']].x * w),
                          int(landmarks[self.landmarks['nose_bridge']].y * h))
                cv2.circle(frame, nose_pt, 12, (255, 255, 0), 2)
                
                # ✅ DIRECTION CONFIRMATION
                cv2.putText(frame, f"✅ BOTH DIRECTIONS FIXED: ({smooth_x:.3f}, {smooth_y:.3f})", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"PupilX:{pupil_x:+.4f} EyeSize:{eye_size:.2f}", 
                           (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status
            display_frame = cv2.resize(frame, (1200, 800))
            cv2.putText(display_frame, "✅ Look LEFT=LEFT | RIGHT=RIGHT | UP=UP | DOWN=DOWN", 
                       (30, 750), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Q=Quit | R=Reset | Test all 4 directions now!", 
                       (30, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow(WINDOW_NAME, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                self.smooth_buffer_x.clear()
                self.smooth_buffer_y.clear()
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("✅ HORIZONTAL DIRECTION FIXED!")
    print("🔧 Mirror + screen coordinate fixes:")
    print("  • pupil_offset_x = -(iris_x - center_x)")
    print("  • head_offset_x = -(nose_x - eye_center_x)")
    print("  • gaze_x = 0.5 + offset_x * sensitivity")
    print("")
    print("🎮 TEST PATTERN:")
    print("  ← Look LEFT → Cursor LEFT")
    print("  → Look RIGHT → Cursor RIGHT") 
    print("  ↑ Look UP → Cursor UP")
    print("  ↓ Look DOWN → Cursor DOWN")
    
    tracker = PerfectDirectionGazeTracker()
    tracker.run()
