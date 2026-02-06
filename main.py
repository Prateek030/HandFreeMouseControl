import threading
import tkinter as tk
from tkinter import ttk
import signal
import sys

# ===================== IMPORT YOUR CLASSES =====================
from Nose_movement_based_trial import DragBlinkTracker
from voice_command_trial import SmartVoiceAssistant

# ===================== MAIN CONTROLLER GUI =====================
class UnifiedHumanInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 Unified Human–Computer Interface")
        self.root.geometry("520x720")
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

        # ===================== HEADER =====================
        title = tk.Label(
            self.root,
            text="Nose Cursor + Voice Command System",
            font=("Segoe UI", 14, "bold"),
            fg="#7FDBFF"
        )
        title.pack(pady=10)

        # ===================== STATUS PANEL =====================
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill="x", padx=10)

        self.nose_status = tk.Label(
            status_frame, text="🟢 Nose Cursor: RUNNING",
            fg="green", font=("Segoe UI", 10)
        )
        self.nose_status.pack(anchor="w")

        self.voice_status = tk.Label(
            status_frame, text="🟢 Voice Assistant: LISTENING",
            fg="green", font=("Segoe UI", 10)
        )
        self.voice_status.pack(anchor="w")

        ttk.Separator(self.root).pack(fill="x", pady=8)

        # ===================== EMBED VOICE UI =====================
        self.voice_container = tk.Frame(self.root)
        self.voice_container.pack(fill="both", expand=True)

        # Inject SmartVoiceAssistant UI into this window
        self.voice_assistant = SmartVoiceAssistant()
        self.voice_assistant.root = self.root
        self.voice_assistant.run = lambda: None  # prevent second mainloop

        # ===================== START SYSTEM THREADS =====================
        self.start_nose_cursor()
        self.start_voice_listener()

        # Handle CTRL+C cleanly
        signal.signal(signal.SIGINT, self.shutdown)

    # ===================== NOSE CURSOR =====================
    def start_nose_cursor(self):
        self.nose_tracker = DragBlinkTracker()
        self.nose_tracker.start()

    # ===================== VOICE ASSISTANT =====================
    def start_voice_listener(self):
        self.voice_thread = threading.Thread(
            target=self.voice_assistant.listen_loop,
            daemon=True
        )
        self.voice_thread.start()

    # ===================== CLEAN SHUTDOWN =====================
    def shutdown(self, *args):
        try:
            self.voice_assistant.listening = True
        except:
            pass

        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    # ===================== RUN =====================
    def run(self):
        self.root.mainloop()


# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    app = UnifiedHumanInterface()
    app.run()
