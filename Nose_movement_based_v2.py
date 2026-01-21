import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
from collections import deque

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

class DragBlinkTracker:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.smooth_vel_x = self.smooth_vel_y = 0.0
        self.prev_nose = None
        
        # DRAG STATE
        self.is_dragging = False
        self.drag_start_time = 0
        
        # BLINK BUFFERS & COOLDOWNS
        self.left_blink_buffer = deque(maxlen=8)
        self.right_blink_buffer = deque(maxlen=8)
        self.left_click_cooldown = self.right_click_cooldown = 0
        
        # CALIBRATION
        self.left_eye_baseline = 25.0
        self.right_eye_baseline = 25.0
        self.calibrated = False
        self.calib_frames = 0
        
        self.sensitivity = 1.0
        self.norm_factor = 0.015
        
        # EYE LANDMARKS
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 373

    def simple_calibration(self, landmarks, h):
        left_height = abs(landmarks[self.LEFT_EYE_TOP].y - landmarks[self.LEFT_EYE_BOTTOM].y) * h
        right_height = abs(landmarks[self.RIGHT_EYE_TOP].y - landmarks[self.RIGHT_EYE_BOTTOM].y) * h
        
        self.left_blink_buffer.append(left_height)
        self.right_blink_buffer.append(right_height)
        self.calib_frames += 1
        
        if self.calib_frames >= 60:
            self.left_eye_baseline = np.mean(self.left_blink_buffer)
            self.right_eye_baseline = np.mean(self.right_blink_buffer)
            self.calibrated = True
            print(f"✅ Calibration COMPLETE!")
            return True
        return False

    def detect_smart_left_blink(self, landmarks, h):
        eye_height = abs(landmarks[self.LEFT_EYE_TOP].y - landmarks[self.LEFT_EYE_BOTTOM].y) * h
        ratio = eye_height / self.left_eye_baseline
        is_blink = ratio < 0.4
        self.left_blink_buffer.append(is_blink)
        return all(list(self.left_blink_buffer)[-4:])

    def detect_smart_right_blink(self, landmarks, h):
        eye_height = abs(landmarks[self.RIGHT_EYE_TOP].y - landmarks[self.RIGHT_EYE_BOTTOM].y) * h
        ratio = eye_height / self.right_eye_baseline
        is_blink = ratio < 0.4
        self.right_blink_buffer.append(is_blink)
        return all(list(self.right_blink_buffer)[-4:])

    def proportional_movement(self, dx_raw, dy_raw, sensitivity):
        dx_norm = np.tanh(dx_raw * self.norm_factor)
        dy_norm = np.tanh(dy_raw * self.norm_factor)
        speed = np.sqrt(dx_raw**2 + dy_raw**2)
        accel_factor = 0.1 + speed * 0.0005
        scale = (0.12 + accel_factor) * sensitivity
        return dx_norm * screen_w * scale, dy_norm * screen_h * scale

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # CALIBRATION PHASE
        print("🔄 CALIBRATION: Look straight for 2 seconds...")
        while not self.calibrated:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                if self.simple_calibration(landmarks, h):
                    break
            
            cv2.rectangle(frame, (w//2-200, h//2-50), (w//2+200, h//2+50), (0, 0, 0), -1)
            cv2.putText(frame, "CALIBRATING...", (w//2-120, h//2+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame {self.calib_frames}/60", (w//2-80, h//2+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('Calibration', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        if not self.calibrated: 
            cap.release()
            cv2.destroyAllWindows()
            return
        
        # TRACKING PHASE
        print("🎮 DRAG MODE ACTIVE!")
        print("👁️ LEFT BLINK + NOSE = DRAG | RIGHT BLINK = RIGHT CLICK")
        print("+/- Sensitivity | Q=Quit")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            dx_raw, dy_raw = 0, 0
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                nose = landmarks[1]
                nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                cv2.circle(frame, (nose_x, nose_y), 8, (0, 255, 0), 2)
                
                # NOSE MOVEMENT
                if self.prev_nose:
                    dx_raw = nose_x - self.prev_nose[0]
                    dy_raw = nose_y - self.prev_nose[1]
                    if abs(dx_raw) > 2 or abs(dy_raw) > 2:
                        cursor_dx, cursor_dy = self.proportional_movement(dx_raw, dy_raw, self.sensitivity)
                        self.smooth_vel_x = 0.75 * cursor_dx + 0.25 * self.smooth_vel_x
                        self.smooth_vel_y = 0.75 * cursor_dy + 0.25 * self.smooth_vel_y
                        
                        # DRAG MODE: Left blink active = drag
                        if self.is_dragging:
                            pyautogui.dragRel(int(self.smooth_vel_x), int(self.smooth_vel_y))
                        else:
                            pyautogui.moveRel(int(self.smooth_vel_x), int(self.smooth_vel_y))
                
                self.prev_nose = (nose_x, nose_y)
                
                # **SMART BLINK DETECTION**
                left_blink = self.detect_smart_left_blink(landmarks, h)
                right_blink = self.detect_smart_right_blink(landmarks, h)
                
                # LEFT BLINK → TOGGLE DRAG MODE
                if left_blink and time.time() - self.left_click_cooldown > 0.6:
                    self.is_dragging = not self.is_dragging
                    if self.is_dragging:
                        pyautogui.mouseDown()  # Start drag
                        self.drag_start_time = time.time()
                    else:
                        pyautogui.mouseUp()    # End drag
                    self.left_click_cooldown = time.time()
                
                # RIGHT BLINK → RIGHT CLICK
                if right_blink and time.time() - self.right_click_cooldown > 0.6:
                    pyautogui.rightClick()
                    self.right_click_cooldown = time.time()
                
                # VISUAL FEEDBACK
                drag_status = "DRAG ON" if self.is_dragging else "DRAG OFF"
                cv2.putText(frame, f"LEFT BLINK: {drag_status}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.is_dragging else (0, 165, 255), 2)
                cv2.putText(frame, "RIGHT BLINK: Right Click", (50, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # CONTROLS
            key = cv2.waitKey(1) & 0xFF
            if key == ord('+') or key == ord('='):
                self.sensitivity = min(3.0, self.sensitivity + 0.1)
                print(f"Sensitivity: {self.sensitivity:.1f}")
            elif key == ord('-'):
                self.sensitivity = max(0.1, self.sensitivity - 0.1)
                print(f"Sensitivity: {self.sensitivity:.1f}")
            elif key == ord('q'):
                break
            
            cv2.putText(frame, f"CALIBRATED | Sens: {self.sensitivity:.1f}", 
                       (50, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('Drag Blink Tracker', frame)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = DragBlinkTracker()
    tracker.run()
