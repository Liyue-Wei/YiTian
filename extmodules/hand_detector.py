# -*- coding: utf-8 -*-
'''
YiTian - Hand Detector Module

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

from multiprocessing import shared_memory
import shm_cfg
import cv2

class HandDetector:
    def __init__(self):
        try:
            shm_frame = shared_memory.SharedMemory(name=shm_cfg.SHM_FRAME_ID)
        except FileNotFoundError:
            raise FileNotFoundError(f"Shared Memory unavaliable: {shm_cfg.SHM_FRAME_ID}")