import speech_recognition as sr
import pyttsx3
import pyautogui
import webbrowser
import os
import subprocess
import wikipedia
import threading
import time
import psutil
from datetime import datetime
import winreg

# Windows fail-safes
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# Initialize Windows TTS (better voices)
tts = pyttsx3.init()
voices = tts.getProperty('voices')
if len(voices) > 1:
    tts.setProperty('voice', voices[1].id)  # Female voice
tts.setProperty('rate', 180)

recognizer = sr.Recognizer()
recognizer.energy_threshold = 1000  # FIXED: Lower sensitivity
recognizer.pause_threshold = 0.8

def speak(text):
    print(f"Assistant: {text}")
    tts.say(text)
    tts.runAndWait()

def listen():
    """FIXED: Proper timeout handling + calibration"""
    try:
        with sr.Microphone() as source:
            print("🔇 Calibrating microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            print("🎤 Listening... (speak within 5 seconds)")
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
        
        command = recognizer.recognize_google(audio).lower()
        print(f"✅ You said: {command}")
        return command
        
    except sr.WaitTimeoutError:
        print("⏰ No speech detected")
        return ""
    except sr.UnknownValueError:
        print("🤷 Didn't understand")
        return ""
    except sr.RequestError:
        print("🌐 Network error")
        speak("Check internet")
        return ""

def execute_command(cmd):
    cmd = cmd.lower()
    
    # Mouse controls
    if any(x in cmd for x in ["mouse left", "click left", "left click"]):
        pyautogui.click()
        speak("Left click")
    elif any(x in cmd for x in ["mouse right", "right click"]):
        pyautogui.rightClick()
        speak("Right click")
    elif "double click" in cmd:
        pyautogui.doubleClick()
        speak("Double click")
    elif "scroll up" in cmd:
        pyautogui.scroll(3)
        speak("Scroll up")
    elif "scroll down" in cmd:
        pyautogui.scroll(-3)
        speak("Scroll down")
    
    # Keyboard
    elif "type" in cmd:
        text = cmd.replace("type", "").strip()
        pyautogui.write(text)
        speak(f"Typed: {text}")
    
    # Windows apps
    elif "notepad" in cmd:
        os.startfile("notepad.exe")
        speak("Notepad opened")
    elif "calculator" in cmd or "calc" in cmd:
        os.startfile("calc.exe")
        speak("Calculator opened")
    elif "paint" in cmd:
        os.startfile("mspaint.exe")
        speak("Paint opened")
    elif "task manager" in cmd:
        os.startfile("taskmgr.exe")
        speak("Task Manager opened")
    
    # Windows volume (FIXED: PowerShell hotkeys)
    elif any(x in cmd for x in ["volume up", "volume increase"]):
        subprocess.run(['powershell', '-Command', 
                       '(Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait(\'{VOLUP}\'))'])
        speak("Volume up")
    elif any(x in cmd for x in ["volume down", "volume decrease"]):
        subprocess.run(['powershell', '-Command', 
                       '(Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait(\'{VOLDOWN}\'))'])
        speak("Volume down")
    elif "mute" in cmd:
        subprocess.run(['powershell', '-Command', 
                       '(Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait(\'{VOLUME_MUTE}\'))'])
        speak("Muted")
    
    # Web
    elif "google" in cmd:
        webbrowser.open("https://google.com")
        speak("Google opened")
    elif "youtube" in cmd:
        webbrowser.open("https://youtube.com")
        speak("YouTube opened")
    
    # Info
    elif "time" in cmd:
        now = datetime.now().strftime("%I:%M %p")
        speak(f"Time is {now}")
    elif "date" in cmd:
        today = datetime.now().strftime("%A, %B %d, %Y")
        speak(f"Today is {today}")
    
    # Shutdown (safety)
    elif any(x in cmd for x in ["shutdown", "shut down"]):
        speak("Shutting down in 10 seconds. Say STOP!")
        time.sleep(5)
        subprocess.run(["shutdown", "/s", "/t", "5"])
    
    elif any(x in cmd for x in ["stop", "exit", "quit"]):
        return False
    
    else:
        speak("Try: click left, notepad, volume up, time")
    
    return True

def main():
    speak("🎙️ Windows Voice Control activated!")
    print("\n🗣️ COMMANDS:")
    print("- Mouse: 'click left', 'right click', 'scroll up'")
    print("- Apps: 'notepad', 'calculator', 'task manager'")
    print("- Volume: 'volume up', 'volume down', 'mute'")
    print("- Web: 'google', 'youtube'")
    print("- Info: 'time', 'date'")
    print("- Exit: 'stop'\n")
    
    while True:
        command = listen()
        if command:
            if not execute_command(command):
                speak("Voice control stopped!")
                break
        else:
            print("🔄 Listening...")
        time.sleep(0.2)

if __name__ == "__main__":
    main()
