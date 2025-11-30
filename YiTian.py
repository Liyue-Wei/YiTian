# -*- coding: utf-8 -*-
'''
YiTian - Touch Typing Correction System

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

from extmodules import shm_cfg
from extmodules import keyboard_listener
from multiprocessing import shared_memory
import cv2
import struct
import threading
import numpy as np
import tkinter as tk

def main():
    kbl = keyboard_listener.KeyboardListener()

    while True:
        # 获取按键（如果没有按键，返回 None；如果有，返回字符）
        key = kbl.get_key()

        if key is not None:
            # --- 在这里写按键触发的逻辑 ---
            if key == 'ESC':
                break  # 退出循环
            elif key == ' ':
                print("用户按下了空格")
            elif key == 'BACKSPACE':
                print("用户删除了一个字符")
            else:
                print(f"用户输入: {key}")

if __name__ == "__main__":
    main()