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
                        case keyboard.Key.enter:
                            self.pressed_key = "ENTER"
                        case keyboard.Key.shift | keyboard.Key.shift_l | keyboard.Key.shift_r:
                            self.pressed_key = "SHIFT"
                        case keyboard.Key.ctrl | keyboard.Key.ctrl_l | keyboard.Key.ctrl_r:
                            self.pressed_key = "CTRL"
                        case keyboard.Key.alt | keyboard.Key.alt_l | keyboard.Key.alt_r | keyboard.Key.alt_gr:
                            self.pressed_key = "ALT"
                        case keyboard.Key.caps_lock:
                            self.pressed_key = "CAPSLOCK"
                        case keyboard.Key.cmd | keyboard.Key.cmd_l | keyboard.Key.cmd_r:
                            self.pressed_key = "CMD"
                        case keyboard.Key.up:
                            self.pressed_key = "UP"
                        case keyboard.Key.down:
                            self.pressed_key = "DOWN"
                        case keyboard.Key.left:
                            self.pressed_key = "LEFT"
                        case keyboard.Key.right:
                            self.pressed_key = "RIGHT"
                        case keyboard.Key.delete:
                            self.pressed_key = "DELETE"
                        case keyboard.Key.insert:
                            self.pressed_key = "INSERT"
                        case keyboard.Key.home:
                            self.pressed_key = "HOME"
                        case keyboard.Key.end:
                            self.pressed_key = "END"
                        case keyboard.Key.page_up:
                            self.pressed_key = "PAGE_UP"
                        case keyboard.Key.page_down:
                            self.pressed_key = "PAGE_DOWN"
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
        
    def stop_listener(self):
        if self.listener:
            self.listener.stop()

'''
from extmodules import keyboard_listener

kbl = keyboard_listener.KeyboardListener()
key = kbl.get_key()
if key != None: print(key)

kbl.stop_listener()
'''