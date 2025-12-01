# -*- coding: utf-8 -*-
'''
YiTian - Touch Typing Correction System

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

from extmodules import shm_cfg
from extmodules import keyboard_listener
import multiprocessing
from multiprocessing import shared_memory
import cv2
import struct
import subprocess
import numpy as np
# import customtkinter as ctk
# import pywinstyles
# from PIL import Image, ImageTk
import tkinter as tk

def camera(num):
    try:
        cam = cv2.VideoCapture(num)
        if not cam.isOpened():
            raise IOError(f"Camera {num} can not be opened...")
        
        res = [shm_cfg.WIDTH, shm_cfg.HEIGHT, shm_cfg.FPS]
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cam.set(cv2.CAP_PROP_FPS, res[2])

        real_res = [int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                    int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                    int(cam.get(cv2.CAP_PROP_FPS))]
        if not real_res == res:
            raise IOError(f"Resolution Setting Unavailable: Expected {res}, but got {real_res}...")
        
        try:
            shm_frame = shared_memory.SharedMemory(create=True, size=shm_cfg.FRAME_SIZE, name=shm_cfg.SHM_FRAME_ID)
        except FileExistsError:
            pass

    except Exception as e:
        print(f"Error: {e}")

def hand_detector():
    pass

def main():
    camera(0)

if __name__ == "__main__":
    main()