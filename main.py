import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os
import signal

PYTHON = sys.executable

CHILD_SCRIPTS = {
    "Nose Cursor + Blink": "MouseControl.py",
    "Voice Assistant": "voice_command.py"
}


class MasterControllerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 Master Human Interface Controller")
        self.root.geometry("480x360")
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

        self.processes = {}
        self.build_ui()

    # ===================== UI =====================
    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Unified Human Control System",
            font=("Segoe UI", 14, "bold"),
            fg="#7FDBFF"
        )
        title.pack(pady=10)

        self.container = tk.Frame(self.root)
        self.container.pack(fill="both", expand=True, padx=15)

        self.rows = {}

        for name in CHILD_SCRIPTS:
            row = self.create_row(self.container, name)
            self.rows[name] = row

        ttk.Separator(self.root).pack(fill="x", pady=10)

        quit_btn = ttk.Button(
            self.root, text="🛑 Quit All", command=self.shutdown
        )
        quit_btn.pack(pady=5)

    def create_row(self, parent, name):
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=6)

        label = tk.Label(frame, text=name, font=("Segoe UI", 11))
        label.pack(side="left")

        status = tk.Label(
            frame, text="STOPPED", fg="red", width=10
        )
        status.pack(side="right")

        start_btn = ttk.Button(
            frame, text="Start",
            command=lambda: self.start_process(name)
        )
        start_btn.pack(side="right", padx=4)

        stop_btn = ttk.Button(
            frame, text="Stop",
            command=lambda: self.stop_process(name)
        )
        stop_btn.pack(side="right", padx=4)

        return status

    # ===================== PROCESS CONTROL =====================
    def start_process(self, name):
        if name in self.processes:
            return

        script = CHILD_SCRIPTS[name]

        if not os.path.exists(script):
            self.rows[name].config(text="MISSING", fg="orange")
            return

        proc = subprocess.Popen(
            [PYTHON, script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if os.name == "nt" else 0
        )

        self.processes[name] = proc
        self.rows[name].config(text="RUNNING", fg="green")

    def stop_process(self, name):
        proc = self.processes.get(name)
        if not proc:
            return

        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        except:
            pass

        proc.kill()
        del self.processes[name]
        self.rows[name].config(text="STOPPED", fg="red")

    # ===================== CLEAN SHUTDOWN =====================
    def shutdown(self):
        for name in list(self.processes.keys()):
            self.stop_process(name)

        self.root.destroy()
        sys.exit(0)

    def run(self):
        self.root.mainloop()


# ===================== ENTRY =====================
if __name__ == "__main__":
    MasterControllerGUI().run()
