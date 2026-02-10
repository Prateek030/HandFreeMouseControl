Below is a **polished, production-quality `README.md`** generated **specifically from your updated files and their actual behavior**:

* `main.py` → **Master GUI launcher**
* `MouseControl.py` → **Nose + blink mouse control**
* `voice_command.py` → **Voice assistant**

This README accurately reflects what the code does today, not aspirational features, and is suitable for **GitHub, portfolio, or research demos**

---

# 🧠 Unified Human–Computer Interaction System

A **hands-free multimodal human–computer interaction system** that combines
**computer vision–based mouse control** and **voice commands**, orchestrated by a **single master GUI**.

This project enables users to control a computer using:

* 🖱️ **Nose movement** for cursor control
* 👁️ **Eye blinks** for mouse clicks
* 🎙️ **Voice commands** for system and application control

All subsystems are launched, monitored, and terminated from **one master control panel**.

---

## ✨ Key Features

### 🎯 Mouse Control (Computer Vision)

* Nose movement → smooth cursor movement
* Left eye blink → left click
* Right eye blink → right click
* Automatic eye-open calibration
* Adaptive acceleration (fast flicks move faster)
* Muted, low-saturation camera UI for reduced eye strain
* Adjustable sensitivity (`+` / `-` keys)

### 🎙️ Voice Command Assistant

* Wake-word based activation
* Intent recognition with confidence scoring
* Application launching (Chrome, VS Code, Explorer, etc.)
* Editing & browser shortcuts (copy, paste, undo, new tab…)
* Context-aware modes (SYSTEM / EDIT / BROWSER)
* Unknown command logging for future learning

### 🪟 Master Control GUI

* Single launcher (`main.py`)
* Start / stop each subsystem independently
* Clean shutdown of child processes
* Prevents camera & microphone conflicts
* Fault-tolerant (each subsystem runs in its own process)

---

## 🏗️ Architecture Overview

```
main.py
│
├── MouseControl.py
│   └── Nose-based cursor + blink detection (OpenCV + MediaPipe)
│
├── voice_command.py
│   └── Voice assistant (SpeechRecognition + rule-based NLP)
│
└── OS-level process isolation (subprocess)
```

Each module runs in a **separate Python process**, ensuring:

* Stability
* No shared event loops
* No device contention
* Easy extensibility

---

## 📁 Project Structure

```
project/
│
├── main.py                 # Master GUI launcher
├── MouseControl.py         # Nose + blink mouse control
├── voice_command.py        # Voice assistant
├── unknown_commands.json   # Auto-generated (voice assistant)
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd project
```

### 2️⃣ Install Dependencies

```bash
pip install opencv-python mediapipe numpy pyautogui
pip install SpeechRecognition keyboard
```

#### Windows (Microphone Support)

If `pyaudio` fails:

```bash
pip install pipwin
pipwin install pyaudio
```

---

### 3️⃣ Run the System

```bash
python main.py
```

This opens the **Master Human Interface Controller**.

From the GUI:

* ▶ Start **Nose Cursor + Blink**
* ▶ Start **Voice Assistant**
* ⏹ Stop either independently
* 🛑 Quit all safely

---

## 🎮 Controls

### 🖱️ Mouse Control (Camera Window)

| Action               | Control         |
| -------------------- | --------------- |
| Cursor move          | Nose movement   |
| Left click           | Left eye blink  |
| Right click          | Right eye blink |
| Increase sensitivity | `+` or `=`      |
| Decrease sensitivity | `-`             |
| Quit mouse control   | `Q`             |

---

### 🎙️ Voice Assistant

**Wake words**

* `computer`
* `assistant`
* `hey system`

**Example commands**

* “computer open chrome”
* “assistant new tab”
* “hey system copy”
* “computer paste”
* “assistant take screenshot”

---

## 🧠 Design Philosophy

* Zero-regression integration
* Hardware isolation (camera & mic)
* Human-centric interaction
* Research-ready structure
* Assistive-technology friendly

This architecture is intentionally designed to support:

* Adaptive learning
* Multimodal intent fusion
* Reinforcement-based personalization
* Accessibility research

---

## 🔮 Future Enhancements

* Voice-controlled cursor modes (precision / fast)
* Learning-based sensitivity adaptation
* Gaze-only fallback (no nose movement)
* Inter-process intent bus (IPC)
* Health monitoring & auto-restart
* Single-EXE packaging

---

## ⚠️ Known Limitations

* Requires stable lighting for face tracking
* Microphone quality affects recognition accuracy
* Single-user calibration per session

---

## 🛡️ Disclaimer

This software controls system-level input (mouse & keyboard).
Use responsibly.
Do **not** run with elevated/admin privileges.

---

## 👤 Author

**Pratik Chopade**
Computer Vision • Human–Computer Interaction • Generative AI

---

If you want next, I can:

* 📦 Package this into a **Windows EXE**
* 🧠 Add **learning-based adaptation**
* 🔊 Let **voice commands control mouse sensitivity**
* 📊 Add **real-time diagnostics in the master GUI**

Just tell me the next step 🚀
