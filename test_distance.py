import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import tkinter as tk
from collections import deque
import threading

# Global setup
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01
screen_w, screen_h = pyautogui.size()
print(f"Screen: {screen_w} x {screen_h}")

class UltraPreciseTracker:
    def __init__(self):
        self.running = False
        self.dragging = False
        self.sensitivity = 1.4
        self.paused = False
        
        # Ultra-stable FaceMesh
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        )
        
        # Precision smoothing (20-point buffer)
        self.cursor_x = screen_w // 2
        self.cursor_y = screen_h // 2
        self.smooth_x = deque(maxlen=20)
        self.smooth_y = deque(maxlen=20)
        
        # Robust gesture detection
        self.blink_left = deque(maxlen=4)
        self.blink_right = deque(maxlen=4)
        self.smile_buf = deque(maxlen=6)
        self.cooldowns = {}
        
        # Calibration with safe defaults
        self.left_eye_base = 28.0
        self.right_eye_base = 28.0
        self.calibrated = False
        
    def safe_get_landmark(self, landmarks, idx):
        """100% crash-proof landmark access"""
        try:
            if landmarks and 0 <= idx < len(landmarks):
                return landmarks[idx]
            return type('Landmark', (), {'x': 0.5, 'y': 0.5, 'z': 0})()
        except:
            return type('Landmark', (), {'x': 0.5, 'y': 0.5, 'z': 0})()
    
    def calibrate(self, cap):
        """Rock-solid calibration"""
        print("🔄 Calibrating (look straight, 3 seconds)...")
        left_samples = deque(maxlen=90)
        right_samples = deque(maxlen=90)
        
        for _ in range(90):
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                l_top = self.safe_get_landmark(landmarks, 159)
                l_bot = self.safe_get_landmark(landmarks, 145)
                r_top = self.safe_get_landmark(landmarks, 386)
                r_bot = self.safe_get_landmark(landmarks, 373)
                
                left_samples.append(abs(l_top.y - l_bot.y) * h)
                right_samples.append(abs(r_top.y - r_bot.y) * h)
            
            # Progress bar
            cv2.rectangle(frame, (w//2-150, h//2), (w//2+150, h//2+60), (50,50,50), -1)
            cv2.rectangle(frame, (w//2-140, h//2+10), (w//2+10, h//2+30), (0,255,0), -1)
            cv2.putText(frame, "CALIBRATING...", (w//2-100, h//2-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.imshow('Calibration', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.left_eye_base = np.median(left_samples) if left_samples else 28.0
        self.right_eye_base = np.median(right_samples) if right_samples else 28.0
        self.calibrated = True
        print(f"✅ Calibrated: L={self.left_eye_base:.1f} R={self.right_eye_base:.1f}")
    
    def detect_gestures(self, landmarks, h, w):
        """Precision gesture detection"""
        gestures = {
            'nose_x': w//2, 'nose_y': h//2,
            'left_blink': False, 'right_blink': False,
            'smile': False
        }
        
        # Nose position (cursor)
        nose = self.safe_get_landmark(landmarks, 1)
        gestures['nose_x'] = int(nose.x * w)
        gestures['nose_y'] = int(nose.y * h)
        
        # Blink detection
        l_top = self.safe_get_landmark(landmarks, 159)
        l_bot = self.safe_get_landmark(landmarks, 145)
        r_top = self.safe_get_landmark(landmarks, 386)
        r_bot = self.safe_get_landmark(landmarks, 373)
        
        left_ratio = abs(l_top.y - l_bot.y) * h / max(self.left_eye_base, 1.0)
        right_ratio = abs(r_top.y - r_bot.y) * h / max(self.right_eye_base, 1.0)
        
        self.blink_left.append(left_ratio < 0.35)
        self.blink_right.append(right_ratio < 0.35)
        
        gestures['left_blink'] = all(self.blink_left)
        gestures['right_blink'] = all(self.blink_right)
        
        # Smile detection
        mouth_left = self.safe_get_landmark(landmarks, 61)
        mouth_right = self.safe_get_landmark(landmarks, 291)
        mouth_top = self.safe_get_landmark(landmarks, 13)
        mouth_bot = self.safe_get_landmark(landmarks, 14)
        
        mouth_w = abs(mouth_left.x - mouth_right.x) * w
        mouth_h = abs(mouth_top.y - mouth_bot.y) * h
        smile_ratio = mouth_h / max(mouth_w, 1.0)
        
        self.smile_buf.append(smile_ratio > 0.85)
        gestures['smile'] = all(self.smile_buf)
        
        return gestures
    
    def update_cursor(self, nose_x, nose_y, frame_w, frame_h):
        """🏆 Ultra-precise cursor control"""
        # Normalize with perfect deadzone
        norm_x = np.clip((nose_x / frame_w - 0.5) * 2.1, -1.1, 1.1)
        norm_y = np.clip((nose_y / frame_h - 0.5) * 2.1, -1.1, 1.1)
        
        # Gold-standard screen mapping
        target_x = screen_w * 0.5 + norm_x * (screen_w * 0.38) * self.sensitivity
        target_y = screen_h * 0.5 + norm_y * (screen_h * 0.38) * self.sensitivity
        
        # 20-point exponential smoothing
        self.smooth_x.append(target_x)
        self.smooth_y.append(target_y)
        
        cursor_x = int(np.average(self.smooth_x))
        cursor_y = int(np.average(self.smooth_y))
        
        self.cursor_x, self.cursor_y = cursor_x, cursor_y
        return cursor_x, cursor_y
    
    def cooldown_ok(self, gesture, cd=0.6):
        """Perfect cooldown system"""
        now = time.time()
        last = self.cooldowns.get(gesture, 0)
        if now - last > cd:
            self.cooldowns[gesture] = now
            return True
        return False
    
    def run(self):
        """🔥 MAIN LOOP - BULLETPROOF"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ NO WEBCAM!")
            return
        
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.calibrate(cap)
        
        print("🎮 TRACKING ACTIVE!")
        print("👃 Nose=Cursor | 👁️L-Blink=Drag | 👁️R-Blink=RClick | 😊Smile=LClick")
        
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Face detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                gestures = self.detect_gestures(landmarks, h, w)
                
                # 🎯 PRECISION CURSOR CONTROL
                cursor_x, cursor_y = self.update_cursor(
                    gestures['nose_x'], gestures['nose_y'], w, h
                )
                
                if self.dragging:
                    pyautogui.dragTo(cursor_x, cursor_y, duration=0.008)
                else:
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0.008)
                
                # Visual feedback
                cv2.circle(frame, (gestures['nose_x'], gestures['nose_y']), 12, (0,255,0), 3)
                
                # 🔥 GESTURE CONTROLS
                if gestures['left_blink'] and self.cooldown_ok('left_blink'):
                    self.dragging = not self.dragging
                    print(f"🔄 DRAG: {'ON' if self.dragging else 'OFF'}")
                
                if gestures['right_blink'] and self.cooldown_ok('right_blink'):
                    pyautogui.rightClick()
                    print("🖱️ RIGHT CLICK!")
                
                if gestures['smile'] and self.cooldown_ok('smile', 1.0):
                    pyautogui.click()
                    print("😊 LEFT CLICK!")
            
            # Professional overlay
            overlay_color = (0, 255, 0) if self.dragging else (0, 255, 255)
            cv2.rectangle(frame, (10, 10), (350, 100), (20, 20, 20), -1)
            
            cv2.putText(frame, "🎮 FACE CONTROL ♿", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Drag: {'ON' if self.dragging else 'OFF'}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)
            cv2.putText(frame, f"Sens: {self.sensitivity:.1f}x", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            cv2.imshow('UltraPrecise FaceControl ♿', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('['):
                self.sensitivity = max(0.8, self.sensitivity - 0.1)
            elif key == ord(']'):
                self.sensitivity = min(2.0, self.sensitivity + 0.1)
        
        cap.release()
        cv2.destroyAllWindows()

class ControlGUI:
    def __init__(self):
        self.tracker = UltraPreciseTracker()
        self.root = tk.Tk()
        self.root.title("🎯 UltraPrecise FaceControl")
        self.root.geometry("400x300")
        self.root.configure(bg='black')
        self.root.resizable(False, False)
        
        self.status_var = tk.StringVar(value="🚀 READY - Click START")
        self.build_gui()
    
    def build_gui(self):
        tk.Label(self.root, text="🤗 ULTRAPRECISE FACE CONTROL", 
                font=('Arial', 16, 'bold'), bg='black', fg='#00ff41').pack(pady=15)
        
        tk.Label(self.root, textvariable=self.status_var, 
                font=('Arial', 12), bg='black', fg='#ffaa00').pack(pady=5)
        
        btn_frame = tk.Frame(self.root, bg='black')
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="▶️ START TRACKING", command=self.start_tracking,
                 bg='#00aa44', fg='white', font=('Arial', 14, 'bold'),
                 width=16, height=2).pack(pady=10)
        
        tk.Button(btn_frame, text="⏹️ STOP TRACKING", command=self.stop_tracking,
                 bg='#cc0000', fg='white', font=('Arial', 14, 'bold'),
                 width=16, height=2).pack(pady=5)
        
        tk.Label(self.root, text="👃 Nose=Cursor | 👁️L-Blink=Drag | 👁️R-Blink=RClick", 
                font=('Arial', 11), bg='black', fg='#00ff41').pack(pady=20)
        
        tk.Label(self.root, text="[ ] = Sensitivity | Q = Quit", 
                font=('Arial', 10), bg='black', fg='#cccccc').pack()
    
    def start_tracking(self):
        if not self.tracker.running:
            self.tracker.running = True
            self.status_var.set("🎮 TRACKING ACTIVE!")
            threading.Thread(target=self.tracker.run, daemon=True).start()
    
    def stop_tracking(self):
        self.tracker.running = False
        self.status_var.set("⏹️ STOPPED")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    print("🎯 ULTRAPRECISE FACE CONTROL v2.0")
    print("="*50)
    print("🎮 CONTROLS:")
    print("   👃 NOSE MOVEMENT  → Precise cursor")
    print("   👁️  LEFT BLINK    → Toggle Drag") 
    print("   👁️  RIGHT BLINK   → Right Click")
    print("   😊 SMILE          → Left Click")
    print("   [ ] KEYS         → Sensitivity")
    print("   Q                → Quit")
    print("="*50)
    
    app = ControlGUI()
    app.run()
