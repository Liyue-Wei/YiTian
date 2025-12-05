# -*- coding: utf-8 -*-
'''
YiTian - Hand Detector Module

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

from multiprocessing import shared_memory
import mediapipe as mp
import numpy as np
import shm_cfg
import cv2
import sys

class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=0, detection_con=0.75, track_con=0.75):
        self.shm_frame = None
        self.shm_result = None

        try:
            self.shm_frame = shared_memory.SharedMemory(name=shm_cfg.SHM_FRAME_ID)
        except FileNotFoundError:
            raise FileNotFoundError(f"Shared Memory unavaliable: {shm_cfg.SHM_FRAME_ID}")
        
        try:
            self.shm_result = shared_memory.SharedMemory(create=True, size=shm_cfg.RESULT_SIZE, name=shm_cfg.SHM_RESULT_ID)
        except FileExistsError:
            print("Process: Shared Memory already exists. Cleaning up...")
            try:
                with shared_memory.SharedMemory(name=shm_cfg.SHM_RESULT_ID) as temp_shm:
                    temp_shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                self.cleanup()
                raise RuntimeError(f"Unexpected error occurred: {e}")
            
            self.shm_result = shared_memory.SharedMemory(create=True, size=shm_cfg.RESULT_SIZE, name=shm_cfg.SHM_RESULT_ID)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(mode, max_hands, model_complexity, detection_con, track_con)
        self.result = None

    def read_img(self):
        shm_buf = self.shm_frame.buf
        if shm_buf[0] == shm_cfg.FLAG_IDLE:
            frame_array = np.ndarray((shm_cfg.HEIGHT, shm_cfg.WIDTH, shm_cfg.CHANNELS), dtype=np.uint8, buffer=shm_buf, offset=1)
            return cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        
        elif shm_buf[0] == shm_cfg.FLAG_EXIT:
            self.cleanup()
            sys.exit(0)
        
        return None
    
    def find_hands(self):
        pass

    def cleanup(self):
        if self.shm_frame:
            self.shm_frame.close()
            self.shm_frame = None

        if self.shm_result:
            self.shm_result.close()
            try:
                self.shm_result.unlink()
            except:
                pass
            self.shm_result = None