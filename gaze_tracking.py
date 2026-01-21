"""
Complete Self-Supervised Gaze Pointer System
- Collects: eye ROI + mouse position pairs
- Trains: CNN autoencoder + gaze regressor
- Deploys: Real-time mouse control
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import pyautogui
import time
import os
from collections import deque
import pickle

# Config
SCREEN_W, SCREEN_H = pyautogui.size()
EYE_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 100
DATA_PATH = 'gaze_data.npz'

class GazeDataset(Dataset):
    def __init__(self, eye_data, mouse_data):
        self.eye_data = torch.FloatTensor(eye_data)
        self.mouse_data = torch.FloatTensor(mouse_data)
    
    def __len__(self):
        return len(self.eye_data)
    
    def __getitem__(self, idx):
        return self.eye_data[idx], self.mouse_data[idx]

class CompleteGazeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: eye image -> features
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU()
        )
        
        # Reconstruction decoder (self-supervision)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)
        )
        
        # Gaze regression head
        self.gaze_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()  # Output normalized [0,1] screen coords
        )
    
    def forward(self, x):
        features = self.encoder(x)
        recon = self.decoder(features)
        gaze = self.gaze_head(features)
        return recon, gaze

def normalize_screen(x, y):
    """Convert screen pixels to normalized [0,1]"""
    return x / SCREEN_W, y / SCREEN_H

def extract_eye_roi(frame, landmarks, eye_indices=[33,7,163,144,145,153,154,155,133]):
    """Extract left eye ROI using MediaPipe landmarks"""
    h, w = frame.shape[:2]
    eye_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                          for i in eye_indices])
    x_min, y_min = np.maximum(eye_points.min(0).astype(int), 0)
    x_max, y_max = np.minimum(eye_points.max(0).astype(int), [w, h])
    
    eye_crop = frame[y_min:y_max, x_min:x_max]
    if eye_crop.size > 0:
        eye_crop = cv2.resize(eye_crop, (EYE_SIZE, EYE_SIZE))
        return cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY) / 255.0
    return np.zeros((EYE_SIZE, EYE_SIZE), dtype=np.float32)

def collect_data(duration_minutes=1, min_samples=100):
    """Fixed data collection with validation and fallback"""
    print(f"Collecting data for {duration_minutes} minutes (need {min_samples} samples)...")
    
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    eye_data_list = []
    mouse_data_list = []
    valid_samples = 0
    start_time = time.time()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Relaxed velocity tracking - store first 100 samples regardless
    first_run = True
    prev_mouse = None
    
    print("Ensure your face is visible and well-lit!")
    while (time.time() - start_time < duration_minutes * 60) and valid_samples < min_samples * 2:
        ret, frame = cap.read()
        if not ret: 
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        mx, my = pyautogui.position()
        norm_x, norm_y = normalize_screen(mx, my)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            eye_img = extract_eye_roi(frame, landmarks)
            
            # Validate eye image
            if eye_img is not None and np.mean(eye_img) > 0.01:  # Not all black
                eye_data_list.append(eye_img[None, None, ...])
                mouse_data_list.append([norm_x, norm_y])
                valid_samples += 1
                
                # Show progress
                if valid_samples % 50 == 0:
                    print(f"Collected {valid_samples} valid samples...")
        
        # Visualize for debugging
        cv2.putText(frame, f'Samples: {valid_samples}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('COLLECTING - Keep face centered', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Collection complete: {valid_samples} samples captured")
    
    if len(eye_data_list) == 0:
        print("ERROR: No valid eye data collected!")
        print("TROUBLESHOOTING:")
        print("1. Ensure good lighting and face visibility")
        print("2. Look directly at camera during collection")
        print("3. Check webcam index (try cv2.VideoCapture(1))")
        return None, None
    
    # Safe concatenation
    eye_data = np.concatenate(eye_data_list, axis=0)
    mouse_data = np.array(mouse_data_list)
    
    np.savez(DATA_PATH, eyes=eye_data, mouse=mouse_data)
    print(f"Saved dataset: {eye_data.shape} eyes, {mouse_data.shape} positions")
    
    return eye_data, mouse_data


def train_model():
    """Train the complete model"""
    if not os.path.exists(DATA_PATH):
        print("No data found. Collecting first...")
        eye_data, mouse_data = collect_data(1)  # 5 min quick collection
    else:
        data = np.load(DATA_PATH)
        eye_data, mouse_data = data['eyes'], data['mouse']
    
    print(f"Training on {len(eye_data)} samples...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CompleteGazeNet().to(device)
    
    dataset = GazeDataset(eye_data, mouse_data)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    recon_loss_fn = nn.MSELoss()
    gaze_loss_fn = nn.L1Loss()
    
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for eyes, poss in loader:
            eyes, poss = eyes.to(device), poss.to(device)
            
            recon, pred_gaze = model(eyes)
            recon_loss = recon_loss_fn(recon, eyes)
            gaze_loss = gaze_loss_fn(pred_gaze, poss)
            loss = recon_loss + 10 * gaze_loss  # Weight gaze more
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_gaze_model.pth')
    
    print("Training complete! Model saved.")

## Real-time Inference & Mouse Control
def run_inference():
    """Load trained model and control mouse with gaze"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CompleteGazeNet()
    model.load_state_dict(torch.load('best_gaze_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Smoothing buffer
    smooth_buffer = deque(maxlen=5)
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                eye_img = extract_eye_roi(frame, landmarks)
                eye_tensor = torch.FloatTensor(eye_img[None, None, ...]).to(device)
                
                recon, gaze_pred = model(eye_tensor)
                pred_x, pred_y = gaze_pred[0].cpu().numpy()
                
                # Smooth predictions
                smooth_buffer.append((pred_x, pred_y))
                avg_x = np.mean([p[0] for p in smooth_buffer])
                avg_y = np.mean([p[1] for p in smooth_buffer])
                
                # Move mouse
                screen_x = int(avg_x * SCREEN_W)
                screen_y = int(avg_y * SCREEN_H)
                pyautogui.moveTo(screen_x, screen_y, duration=0.05)
                
                # Visualize prediction
                cv2.circle(frame, (screen_x, screen_y), 10, (0,255,0), 2)
            
            cv2.imshow('Gaze Pointer', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()                   
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    # train_model()
    run_inference()

    # if len(sys.argv) > 1 and sys.argv[1] == 'train':
    #     train_model()
    # elif len(sys.argv) > 1 and sys.argv[1] == 'demo':
    #     run_inference()
    # else:
    #     print("Usage: python gaze_model.py [train|demo]")
    #     print("First run: python gaze_model.py train")
    #     print("Then demo: python gaze_model.py demo")
