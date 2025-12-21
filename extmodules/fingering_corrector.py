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

        self.anchor_key = ['q', 'p', 'z', 'm']
        self.ak_coord = {}
        self.ak_idx = 0
        self.is_calibrated = False

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

    def key_map_calibration(self, pressed_key, finger_pos):
        if not finger_pos:
            return False
        
        if self.ak_idx >= len(self.anchor_key):
            return False
        
        target_key = self.anchor_key[self.ak_idx]
        if pressed_key == target_key:
            pos_tuple = (finger_pos['x'], finger_pos['y'])
            print(f"Process: Captured {target_key} at {pos_tuple}")
            self.ak_coord[target_key] = pos_tuple
            self.ak_idx += 1

            if self.ak_idx == len(self.anchor_key):
                success = self._generate_key_map()
                if success:
                    self.is_calibrated = True
                    print("Process: Calibration successed. Keyboard map generated.")
                    return True
                else:
                    print("Process: Calibration failed. Resetting...")
                    self.reset_calibration()
        
        return False
    
    def reset_calibration(self):
        self.ak_idx = 0
        self.ak_coord = {}
        self.is_calibrated = False
        self.key_map = {}

    def _generate_key_map(self):
        try:
            src_pts = np.float32([
                [0, 0],      # q
                [9, 0],      # p
                [0.75, 2],   # z
                [6.75, 2]    # m
            ])

            dst_pts = np.float32([
                self.ak_coord['q'],
                self.ak_coord['p'],
                self.ak_coord['z'],
                self.ak_coord['m']
            ])

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.key_map = {}
            key_ideal_coords = {
                "qwertyuiop": [(i, 0) for i in range(10)],
                "asdfghjkl": [(i + 0.25, 1) for i in range(9)],
                "zxcvbnm": [(i + 0.75, 2) for i in range(7)]
            }

            for row_str, coords in key_ideal_coords.items():
                ideal_pts = np.float32(coords).reshape(-1, 1, 2)
                transformed_pts = cv2.perspectiveTransform(ideal_pts, matrix)
                for i, char in enumerate(row_str):
                    self.key_map[char] = tuple(transformed_pts[i][0].astype(int))
            
            return True
            
        except Exception as e:
            print(f"Process: Unexpected error occurred: {e}")
            return False

    def check_fingering(self, typed_key, hands_data):
        if not typed_key or typed_key not in self.finger_map: 
            return "No rule", None

        if typed_key not in self.key_map:
            return None, None
            
        correct_finger_name = self.finger_map[typed_key]
        target_key_pos = self.key_map[typed_key]      
        expected_hand_label = "Left" if correct_finger_name.startswith("LEFT") else "Right"
        expected_finger_idx = self.fingertip_indices[correct_finger_name]
        for hand in hands_data:
            if hand['label'] == expected_hand_label:
                lm = hand['landmarks'][expected_finger_idx]
                dist = np.hypot(lm['x'] - target_key_pos[0], lm['y'] - target_key_pos[1])
                if dist < 30: 
                    return "Correct", correct_finger_name

        min_dist = float('inf')
        actual_finger_name = "Unknown"
        for hand in hands_data:
            hand_prefix = "LEFT" if hand['label'] == "Left" else "RIGHT"
            for fname, idx in self.fingertip_indices.items():
                if not fname.startswith(hand_prefix): 
                    continue
                    
                lm = hand['landmarks'][idx]
                dist = np.hypot(lm['x'] - target_key_pos[0], lm['y'] - target_key_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    actual_finger_name = fname

        if min_dist > 60:
            return "Unknown", "Hand too far"
            
        if actual_finger_name == correct_finger_name:
            return "Correct", correct_finger_name
        
        return "Wrong", actual_finger_name[6:]