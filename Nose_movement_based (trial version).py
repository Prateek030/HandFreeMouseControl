import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
from collections import deque

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# ===================== MUTED COLOR PALETTE =====================
BG_PANEL = (28, 28, 28)
TEXT = (230, 230, 230)
SOFT_GREEN = (120, 200, 170)
SOFT_RED = (110, 90, 200)
SOFT_AMBER = (80, 190, 230)
SOFT_CYAN = (180, 200, 200)
DIM_GREY = (90, 90, 90)

# ===================== IMAGE TUNING =====================
SATURATION_SCALE = 0.45   # <--- reduce saturation
VALUE_SCALE = 1.05        # slight brightness boost

# ===================== MAIN CLASS =====================
class DragBlinkTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True
        )

        self.prev_nose = None
        self.smooth_x = 0.0
        self.smooth_y = 0.0

        self.left_click_time = 0
        self.right_click_time = 0

        self.left_buffer = deque(maxlen=8)
        self.right_buffer = deque(maxlen=8)

        self.calibrated = False
        self.calib_frames = 0
        self.left_open = None
        self.right_open = None

        self.sensitivity = 1.0
        self.norm_factor = 0.015

        # Eye landmarks
        self.LT, self.LB, self.LL, self.LR = 159, 145, 33, 133
        self.RT, self.RB, self.RL, self.RR = 386, 373, 362, 263

    # ===================== FRAME TONE =====================
    def reduce_saturation(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= SATURATION_SCALE
        hsv[..., 2] *= VALUE_SCALE
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ===================== CALIBRATION =====================
    def calibrate(self, lm, h, w):
        lh = abs(lm[self.LT].y - lm[self.LB].y) * h
        lw = abs(lm[self.LL].x - lm[self.LR].x) * w
        rh = abs(lm[self.RT].y - lm[self.RB].y) * h
        rw = abs(lm[self.RL].x - lm[self.RR].x) * w

        self.left_buffer.append(lh / (lw + 1e-6))
        self.right_buffer.append(rh / (rw + 1e-6))
        self.calib_frames += 1

        if self.calib_frames >= 60:
            self.left_open = np.mean(self.left_buffer)
            self.right_open = np.mean(self.right_buffer)
            self.calibrated = True

    # ===================== MOVEMENT =====================
    def move_cursor(self, dx, dy):
        dxn = np.tanh(dx * self.norm_factor)
        dyn = np.tanh(dy * self.norm_factor)
        speed = np.sqrt(dx * dx + dy * dy)
        accel = 0.12 + speed * 0.001

        mx = dxn * screen_w * accel * self.sensitivity
        my = dyn * screen_h * accel * self.sensitivity

        self.smooth_x = 0.75 * mx + 0.25 * self.smooth_x
        self.smooth_y = 0.75 * my + 0.25 * self.smooth_y

        pyautogui.moveRel(int(self.smooth_x), int(self.smooth_y))

    # ===================== UI =====================
    def draw_panel(self, frame):
        cv2.rectangle(frame, (0, 0), (360, 160), BG_PANEL, -1)
        cv2.putText(frame, "Nose Cursor + Blink Control",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SOFT_CYAN, 2)
        cv2.putText(frame, f"Sensitivity: {self.sensitivity:.1f}",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 1)

        status = "CALIBRATING" if not self.calibrated else "ACTIVE"
        color = SOFT_AMBER if not self.calibrated else SOFT_GREEN
        cv2.putText(frame, f"Status: {status}",
                    (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_eye_bar(self, frame, x, y, ratio, base, label):
        pct = min(ratio / base, 1.2)
        bar = int(120 * pct)
        color = SOFT_GREEN if pct > 0.55 else SOFT_RED

        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1)
        cv2.rectangle(frame, (x, y), (x + 120, y + 10), DIM_GREY, 1)
        cv2.rectangle(frame, (x, y), (x + bar, y + 10), color, -1)

    # ===================== MAIN =====================
    def run(self):
        cap = cv2.VideoCapture(0)
        print("🔄 Calibrating — keep eyes open")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = self.reduce_saturation(frame)

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                nx, ny = int(lm[1].x * w), int(lm[1].y * h)
                cv2.circle(frame, (nx, ny), 5, SOFT_GREEN, -1)

                lh = abs(lm[self.LT].y - lm[self.LB].y) * h
                lw = abs(lm[self.LL].x - lm[self.LR].x) * w
                rh = abs(lm[self.RT].y - lm[self.RB].y) * h
                rw = abs(lm[self.RL].x - lm[self.RR].x) * w

                l_ratio = lh / (lw + 1e-6)
                r_ratio = rh / (rw + 1e-6)

                if not self.calibrated:
                    self.calibrate(lm, h, w)
                else:
                    if self.prev_nose:
                        self.move_cursor(nx - self.prev_nose[0],
                                         ny - self.prev_nose[1])

                    now = time.time()
                    if l_ratio < self.left_open * 0.45 and now - self.left_click_time > 0.5:
                        pyautogui.leftClick()
                        self.left_click_time = now
                    if r_ratio < self.right_open * 0.45 and now - self.right_click_time > 0.5:
                        pyautogui.rightClick()
                        self.right_click_time = now

                self.prev_nose = (nx, ny)

                self.draw_panel(frame)
                if self.calibrated:
                    self.draw_eye_bar(frame, 15, 120, l_ratio, self.left_open, "Left Eye")
                    self.draw_eye_bar(frame, 15, 145, r_ratio, self.right_open, "Right Eye")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('+'), ord('=')):
                self.sensitivity = min(3.0, self.sensitivity + 0.1)
            elif key == ord('-'):
                self.sensitivity = max(0.2, self.sensitivity - 0.1)
            elif key == ord('q'):
                break

            cv2.imshow("Smart Nose Cursor", frame)

        cap.release()
        cv2.destroyAllWindows()

# ===================== RUN =====================
if __name__ == "__main__":
    DragBlinkTracker().run()
