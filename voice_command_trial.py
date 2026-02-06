import speech_recognition as sr
import keyboard
import pyautogui
import tkinter as tk
from tkinter import ttk
import threading
import json
import time
import difflib
from datetime import datetime

# ===================== CONFIG =====================
WAKE_WORDS = ["computer", "assistant", "hey system"]
UNKNOWN_LOG_FILE = "unknown_commands.json"

# ===================== APPLICATION ALIASES =====================
APP_ALIASES = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "browser": "chrome",
    "edge": "edge",
    "vscode": "visual studio code",
    "code": "visual studio code",
    "visual studio code": "visual studio code",
    "notepad": "notepad",
    "calculator": "calculator",
    "cmd": "command prompt",
    "terminal": "command prompt",
    "explorer": "file explorer",
    "files": "file explorer",
    "settings": "settings",
}

# ===================== INTENTS =====================
INTENTS = {
    "COPY": {
        "phrases": ["copy", "make a copy", "duplicate"],
        "action": lambda: keyboard.press_and_release("ctrl+c"),
        "mode": "EDIT"
    },
    "PASTE": {
        "phrases": ["paste", "insert here"],
        "action": lambda: keyboard.press_and_release("ctrl+v"),
        "mode": "EDIT"
    },
    "CUT": {
        "phrases": ["cut", "remove selection"],
        "action": lambda: keyboard.press_and_release("ctrl+x"),
        "mode": "EDIT"
    },
    "UNDO": {
        "phrases": ["undo", "go back"],
        "action": lambda: keyboard.press_and_release("ctrl+z"),
        "mode": "EDIT"
    },
    "NEW_TAB": {
        "phrases": ["new tab", "open new tab"],
        "action": lambda: keyboard.press_and_release("ctrl+t"),
        "mode": "BROWSER"
    },
    "ADDRESS_BAR": {
        "phrases": ["address bar", "search bar", "url bar"],
        "action": lambda: keyboard.press_and_release("ctrl+l"),
        "mode": "BROWSER"
    },
    "SEARCH": {
        "phrases": ["search", "find", "look up"],
        "action": "SEARCH_QUERY",
        "mode": "BROWSER"
    },
    "OPEN_APP": {
        "phrases": ["open", "launch", "start"],
        "action": "OPEN_APPLICATION",
        "mode": "SYSTEM"
    },
    "SCREENSHOT": {
        "phrases": ["screenshot", "take screenshot"],
        "action": lambda: keyboard.press_and_release("win+shift+s"),
        "mode": "SYSTEM"
    },
}

# ===================== CONTEXT =====================
class Context:
    def __init__(self):
        self.mode = "SYSTEM"
        self.last_intent = None

context = Context()

# ===================== INTENT RESOLUTION =====================
def resolve_intent(command):
    best_intent = None
    best_score = 0

    for intent, data in INTENTS.items():
        for phrase in data["phrases"]:
            score = difflib.SequenceMatcher(None, phrase, command).ratio()
            if score > best_score:
                best_score = score
                best_intent = intent

    return best_intent, best_score

# ===================== UNKNOWN COMMAND LOGGER =====================
def log_unknown(command):
    try:
        data = []
        try:
            with open(UNKNOWN_LOG_FILE, "r") as f:
                data = json.load(f)
        except:
            pass

        data.append({
            "command": command,
            "time": datetime.now().isoformat()
        })

        with open(UNKNOWN_LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except:
        pass

# ===================== APP LAUNCHER =====================
def open_application(spoken_text):
    app_name = spoken_text
    for alias, real_name in APP_ALIASES.items():
        if alias in spoken_text:
            app_name = real_name
            break

    keyboard.press_and_release("win")
    time.sleep(0.4)
    pyautogui.write(app_name, interval=0.05)
    time.sleep(0.4)
    pyautogui.press("enter")

# ===================== UI + ASSISTANT =====================
class SmartVoiceAssistant:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Voice Assistant")
        self.root.geometry("500x650")
        self.root.attributes("-topmost", True)

        self.status = tk.Label(
            self.root, text="🧠 Say wake word...", font=("Arial", 12), fg="cyan"
        )
        self.status.pack(pady=10)

        self.info = tk.Label(self.root, text="Mode: SYSTEM", font=("Arial", 10))
        self.info.pack()

        self.tree = ttk.Treeview(
            self.root, columns=("Intent", "Phrases", "Mode"), show="headings"
        )
        self.tree.heading("Intent", text="Intent")
        self.tree.heading("Phrases", text="Example Phrases")
        self.tree.heading("Mode", text="Mode")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        for intent, data in INTENTS.items():
            self.tree.insert(
                "", "end",
                values=(intent, ", ".join(data["phrases"]), data["mode"])
            )

        self.listening = True
        threading.Thread(target=self.listen_loop, daemon=True).start()
        keyboard.add_hotkey("esc", self.stop)

    # ===================== VOICE LOOP =====================
    def listen_loop(self):
        r = sr.Recognizer()
        mic = sr.Microphone()

        while self.listening:
            try:
                with mic as source:
                    r.adjust_for_ambient_noise(source, duration=0.6)
                    self.status.config(text="🎤 Listening...", fg="cyan")
                    audio = r.listen(source, timeout=5, phrase_time_limit=5)

                speech = r.recognize_google(audio).lower().strip()
                print("Heard:", speech)

                if not any(w in speech for w in WAKE_WORDS):
                    continue

                self.process_command(speech)

            except sr.UnknownValueError:
                self.status.config(text="❓ Didn't catch that", fg="orange")
            except Exception as e:
                self.status.config(text=str(e), fg="red")

    # ===================== COMMAND PROCESSING =====================
    def process_command(self, speech):
        for w in WAKE_WORDS:
            speech = speech.replace(w, "")
        speech = speech.strip()

        intent, confidence = resolve_intent(speech)

        if intent and confidence > 0.55:
            data = INTENTS[intent]
            context.mode = data["mode"]
            context.last_intent = intent

            if data["action"] == "SEARCH_QUERY":
                keyboard.press_and_release("ctrl+l")
                time.sleep(0.2)
                pyautogui.write(
                    speech.replace("search", "").replace("find", "").strip()
                )
                pyautogui.press("enter")

            elif data["action"] == "OPEN_APPLICATION":
                open_application(speech)

            else:
                data["action"]()

            self.status.config(
                text=f"✅ {intent} ({confidence:.2f})", fg="green"
            )
            self.info.config(text=f"Mode: {context.mode}")

        else:
            log_unknown(speech)
            self.status.config(text="⚠️ Unknown command logged", fg="orange")

    def stop(self):
        self.listening = False
        self.root.quit()

    def run(self):
        self.root.mainloop()

# ===================== RUN =====================
# if __name__ == "__main__":
#     SmartVoiceAssistant().run()
