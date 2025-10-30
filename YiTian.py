import cv2
import mediapipe as mp
import numpy as np
import time
from pynput import keyboard
import tkinter as tk
import threading

class KeyboardListener:
    def __init__(self):
        self.last_key = None
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        with self.lock:
            try:
                if hasattr(key, 'char') and key.char:
                    self.last_key = key.char.lower()
                elif key == keyboard.Key.space:
                    self.last_key = ' '
                elif key == keyboard.Key.enter:
                    self.last_key = '\n'
                elif key == keyboard.Key.tab:
                    self.last_key = '\t'
                elif key == keyboard.Key.backspace:
                    self.last_key = 'BACKSPACE'
                elif key == keyboard.Key.esc:
                    self.last_key = 'ESC'
            except AttributeError:
                pass

    def get_last_key(self):
        with self.lock:
            key = self.last_key
            self.last_key = None
            return key