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
        
class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=0, detection_con=0.3, track_con=0.3):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_con, self.track_con)
        self.results = None
        self.prev_landmarks = None 

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results and self.results.multi_hand_landmarks:
            if self.prev_landmarks:
                alpha = 0.7  
                for i, hand in enumerate(self.results.multi_hand_landmarks):
                    for j, lm in enumerate(hand.landmark):
                        prev_lm = self.prev_landmarks[i].landmark[j]
                        lm.x = alpha * lm.x + (1 - alpha) * prev_lm.x
                        lm.y = alpha * lm.y + (1 - alpha) * prev_lm.y
                        lm.z = alpha * lm.z + (1 - alpha) * prev_lm.z
            
            self.prev_landmarks = self.results.multi_hand_landmarks
        
        return self.results
    
