import speech_recognition as sr
import keyboard
import pyautogui
import tkinter as tk
from tkinter import ttk
import threading
import json
import sys

# Your provided shortcuts as a dict for easy mapping (voice phrase -> shortcut keys)
SHORTCUTS = {
    # Critical Shortcuts
    "copy": "ctrl+c",
    "paste": "ctrl+v",
    "cut": "ctrl+x",
    "undo": "ctrl+z",
    "redo": "ctrl+y",
    "select all": "ctrl+a",
    "save": "ctrl+s",
    
    # Window & App Control
    "close window": "alt+f4",
    "close tab": "ctrl+w",
    "switch apps": "alt+tab",
    "new tab": "ctrl+t",
    "minimize all": "win+d",
    "task manager": "ctrl+shift+esc",
    
    # File Explorer
    "open explorer": "win+e",
    "new folder": "ctrl+shift+n",
    "rename": "f2",
    "refresh": "f5",
    
    # System Control
    "lock pc": "win+l",
    "search": "win+s",
    "settings": "win+i",
    "run dialog": "win+r",
    "volume up": "f11",  # Fn varies; adjust per laptop
    "volume down": "f12",
    "brightness up": "f6",
    "brightness down": "f5",
    
    # Browser Specific
    "new tab": "ctrl+t",  # Duplicate handled
    "address bar": "ctrl+l",
    "reload": "ctrl+r",
    "fullscreen": "f11",
    
    # Navigation Mastery (simulated via pyautogui for reliability)
    "next": lambda: pyautogui.press('tab'),
    "previous": lambda: pyautogui.hotkey('shift', 'tab'),
    "click": lambda: pyautogui.press('enter'),
    "select": lambda: pyautogui.press('space'),
    "cancel": "esc",
    
    # Power User Combos
    "minimize all windows": "win+m",
    "screenshot": "win+shift+s",
    "reopen tab": "ctrl+shift+t",
    "task view": "win+tab",
    "security options": "ctrl+alt+del",
    
    # Voice control toggles
    "stop listening": None  # Exits loop
}

class VoiceController:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice Shortcuts Panel")
        self.root.geometry("400x600")
        self.root.attributes('-topmost', True)
        self.root.configure(bg='black')
        self.root.resizable(False, False)
        
        # Status label
        self.status = tk.Label(self.root, text="Listening... Say a command!", fg='green', bg='black', font=('Arial', 12))
        self.status.pack(pady=10)
        
        # Treeview for shortcuts table
        columns = ("Command", "Shortcut", "Category")
        self.tree = ttk.Treeview(self.root, columns=columns, show='headings', height=25)
        self.tree.heading("Command", text="Voice Command")
        self.tree.heading("Shortcut", text="Shortcut")
        self.tree.heading("Category", text="Category")
        self.tree.column("Command", width=150)
        self.tree.column("Shortcut", width=100)
        self.tree.column("Category", width=120)
        self.tree.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Populate table with your shortcuts
        self.populate_table()
        
        # Start voice listener in thread
        self.listening = True
        self.thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.thread.start()
        
        # Bind ESC to quit
        keyboard.add_hotkey('esc', self.stop_listening)
        
    def populate_table(self):
        categories = {
            "Critical": ["copy", "paste", "cut", "undo", "redo", "select all", "save"],
            "Window": ["close window", "close tab", "switch apps", "new tab", "minimize all", "task manager"],
            "Explorer": ["open explorer", "new folder", "rename", "refresh"],
            "System": ["lock pc", "search", "settings", "run dialog", "volume up", "volume down", "brightness up", "brightness down"],
            "Browser": ["address bar", "reload", "fullscreen"],
            "Navigation": ["next", "previous", "click", "select", "cancel"],
            "Power": ["minimize all windows", "screenshot", "reopen tab", "task view", "security options"]
        }
        for cat_items in categories.values():
            for cmd in cat_items:
                shortcut = SHORTCUTS[cmd]
                if callable(shortcut):
                    shortcut_str = shortcut.__name__
                else:
                    shortcut_str = shortcut
                self.tree.insert("", "end", values=(cmd.title(), shortcut_str, cat_items[0].split()[0].title()))
    
    def listen_loop(self):
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            r.adjust_for_ambient_noise(source)
        
        while self.listening:
            try:
                with mic as source:
                    audio = r.listen(source, timeout=1, phrase_time_limit=3)
                command = r.recognize_google(audio).lower().strip()
                self.status.config(text=f"Heard: {command}", fg='yellow')
                self.execute_command(command)
            except sr.UnknownValueError:
                pass  # Ignore unclear speech
            except sr.RequestError:
                self.status.config(text="Speech service error", fg='red')
            except Exception as e:
                self.status.config(text=f"Error: {str(e)}", fg='red')
    
    def execute_command(self, command):
        for voice_cmd, shortcut in SHORTCUTS.items():
            if voice_cmd in command:
                if callable(shortcut):
                    shortcut()
                elif shortcut:
                    keyboard.press_and_release(shortcut)
                self.status.config(text=f"Executed: {voice_cmd}", fg='green')
                return
        self.status.config(text="Command not found", fg='orange')
    
    def stop_listening(self):
        self.listening = False
        self.root.quit()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = VoiceController()
    app.run()
