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
# import customtkinter as ctk
# import pywinstyles
# from PIL import Image, ImageTk
import tkinter as tk

def capture(num):
    try:
        cap = cv2.VideoCapture(num)
        if not cap.isOpened():
            raise IOError(f"Camera {num} can not be opened...")
    except Exception as e:
        print(f"Error: {e}")

    res = f"{shm_cfg.WIDTH}x{shm_cfg.HEIGHT}"

def main():
    capture(2)

if __name__ == "__main__":
    main()