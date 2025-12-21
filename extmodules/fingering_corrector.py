# -*- coding: utf-8 -*-
'''
YiTian - Fingering Corrector Module

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

from multiprocessing import shared_memory
from extmodules import shm_cfg
import numpy as np
import cv2

class FingeringCorrector:
    def __init__(self):
        self.shm_frame = None
        self.shm_result = None
        self.WIDTH = shm_cfg.WIDTH
        self.HEIGHT = shm_cfg.HEIGHT
        self.float_arr_len = (shm_cfg.RESULT_SIZE - 4) // 4
        self.key_layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        self.key_map = {}
        self.finger_map = {
            'q': 'LEFT_PINKY', 'a': 'LEFT_PINKY', 'z': 'LEFT_PINKY', 
            'w': 'LEFT_RING', 's': 'LEFT_RING', 'x': 'LEFT_RING',
            'e': 'LEFT_MIDDLE', 'd': 'LEFT_MIDDLE', 'c': 'LEFT_MIDDLE', 
            'r': 'LEFT_INDEX', 'f': 'LEFT_INDEX', 'v': 'LEFT_INDEX',
            't': 'LEFT_INDEX', 'g': 'LEFT_INDEX', 'b': 'LEFT_INDEX', 
            
            'y': 'RIGHT_INDEX', 'h': 'RIGHT_INDEX', 'n': 'RIGHT_INDEX',
            'u': 'RIGHT_INDEX', 'j': 'RIGHT_INDEX', 'm': 'RIGHT_INDEX', 
            'i': 'RIGHT_MIDDLE', 'k': 'RIGHT_MIDDLE', 
            'o': 'RIGHT_RING', 'l': 'RIGHT_RING', 
            'p': 'RIGHT_PINKY',
        }

        self.fingertip_indices = {
            "RIGHT_THUMB": 4, "RIGHT_INDEX": 8, "RIGHT_MIDDLE": 12, "RIGHT_RING": 16, "RIGHT_PINKY": 20,
            "LEFT_THUMB": 4, "LEFT_INDEX": 8, "LEFT_MIDDLE": 12, "LEFT_RING": 16, "LEFT_PINKY": 20
        } 

        try:
            self.shm_result = shared_memory.SharedMemory(name=shm_cfg.SHM_RESULT_ID)
            print("Process: Shared Memory Result connected.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Shared Memory unavaliable: {shm_cfg.SHM_RESULT_ID}")      

    def read_shm_data(self):
        if self.shm_result is None:
            return []
        
        try:
            shm_buf = self.shm_result.buf
            count = shm_buf[0]
            if count == 0:
                return []
            
            data_array = np.ndarray((self.float_arr_len,), dtype=np.float32, buffer=shm_buf, offset=4)
            hands = []
            for i in range(count):
                interval = i * 65
                label = "Left" if (data_array[interval] == 1.0) else "Right"
                lm_flat = data_array[interval + 2 : interval + 65]
                landmarks = []
                for j in range(0, 63, 3):
                    landmarks.append({
                        'x': int(lm_flat[j] * self.WIDTH),
                        'y': int(lm_flat[j+1] * self.HEIGHT),
                        'z': lm_flat[j+2]
                    })
                
                hands.append({
                    'label': label,
                    'landmarks': landmarks
                })
                
            return hands
        
        except Exception as e:
            raise(f"Error: Unexpected error occurred: {e}")

    def get_pressing_key(self, landmark, width, height):
        if not landmark: 
            return None

    def key_map_calibration(self, key):
        anchor_key = ['q', 'p', 'z', 'm']