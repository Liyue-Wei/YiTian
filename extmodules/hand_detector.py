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
        self.shm_frame = None
        self.shm_result = None

        try:
            shm_frame = shared_memory.SharedMemory(name=shm_cfg.SHM_FRAME_ID)
        except FileNotFoundError:
            raise FileNotFoundError(f"Shared Memory unavaliable: {shm_cfg.SHM_FRAME_ID}")
        
        try:
            shm_result = shared_memory.SharedMemory(create=True, size=shm_cfg.RESULT_SIZE, name=shm_cfg.SHM_RESULT_ID)
        except FileExistsError:
            print("Process: Shared Memory already exists. Cleaning up...")
            with shared_memory.SharedMemory(name=shm_cfg.SHM_RESULT_ID) as temp_shm:
                temp_shm.unlink()
            shm_result = shared_memory.SharedMemory(create=True, size=shm_cfg.RESULT_SIZE, name=shm_cfg.SHM_RESULT_ID)

    def cleanup():
        pass