# -*- coding: utf-8 -*-
'''
YiTian - Keyboard Listener Module

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
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
            try:
                if hasattr(key, 'char') and key.char:
                    self.pressed_key = key.char.lower()
                else:
                    match key:
                        case keyboard.Key.space:
                            self.pressed_key = ' '
                        case keyboard.Key.enter:
                            self.pressed_key = '\n'
                        case keyboard.Key.tab:
                            self.pressed_key = '\t'
                        case keyboard.Key.backspace:
                            self.pressed_key = "BACKSPACE"
                        case keyboard.Key.esc:
                            self.pressed_key = "ESC"
                        case _:
                            pass    # "Invalid" if needed

            except AttributeError:
                print(f"[Warning] Invalid key attribute: {key}")
                pass    # "Error" if needed

    def get_key(self):
        with self.lock:
            key = self.pressed_key
            self.pressed_key = None
            return key
        
'''
from extmodules import keyboard_listener

kbl = keyboard_listener.KeyboardListener()

key = kbl.get_key()
'''