# -*- coding: utf-8 -*-
'''
YiTian - Keyboard Listener Module

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the MIT License. 
'''

import threading
from pynput import keyboard

class KeyboardListener:
    def __init__(self):
        self.pressed_key = None
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        with self.lock:
            pass