# -*- coding: utf-8 -*-
'''
YiTian - Touch Typing Correction System

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

import cv2
import mediapipe as mp
import numpy as np  # 新增:需要用於透視變換
import time
from pynput import keyboard
import threading
import copy  

class KeyboardListener:
    def __init__(self):
        self.last_key = None
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        with self.lock:
            try:
                if hasattr(key, 'char') and key.char:
                    self.last_key = key.char.lower()
                elif key == keyboard.Key.space:
                    self.last_key = ' '
                elif key == keyboard.Key.enter:
                    self.last_key = '\n'
                elif key == keyboard.Key.tab:
                    self.last_key = '\t'
                elif key == keyboard.Key.backspace:
                    self.last_key = 'BACKSPACE'
                elif key == keyboard.Key.esc:
                    self.last_key = 'ESC'
            except AttributeError:
                pass

    def get_last_key(self):
        with self.lock:
            key = self.last_key
            self.last_key = None
            return key
        
class HandDetector:
    def __init__(self, mode, max_hands, model_complexity, detection_con, track_con):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(mode, max_hands, model_complexity, detection_con, track_con)
        self.results = None
        self.prev_landmarks = None 

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results and self.results.multi_hand_landmarks:
            current_hand_count = len(self.results.multi_hand_landmarks)
            
            if self.prev_landmarks and len(self.prev_landmarks) == current_hand_count:
                alpha = 0.5
                try:
                    for i in range(current_hand_count):
                        hand = self.results.multi_hand_landmarks[i]
                        prev_hand = self.prev_landmarks[i]
                        for j in range(len(hand.landmark)):
                            lm = hand.landmark[j]
                            prev_lm = prev_hand.landmark[j]
                            lm.x = alpha * lm.x + (1 - alpha) * prev_lm.x
                            lm.y = alpha * lm.y + (1 - alpha) * prev_lm.y
                            lm.z = alpha * lm.z + (1 - alpha) * prev_lm.z
                except (IndexError, AttributeError) as e:
                    print(f"平滑處理錯誤: {e}")
                    pass
            
            self.prev_landmarks = copy.deepcopy(self.results.multi_hand_landmarks)
        else:
            self.prev_landmarks = None
        
        return self.results

class TypingCorrector:
    def __init__(self):
        self.key_layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        self.key_map = {}
        self.finger_map = self._generate_finger_map()
        self.fingertip_indices = {
            "RIGHT_THUMB": 4, "RIGHT_INDEX": 8, "RIGHT_MIDDLE": 12, "RIGHT_RING": 16, "RIGHT_PINKY": 20,
            "LEFT_THUMB": 4, "LEFT_INDEX": 8, "LEFT_MIDDLE": 12, "LEFT_RING": 16, "LEFT_PINKY": 20
        }

    def generate_key_map_from_anchors(self, anchor_points):
        """使用4個錨點(Q/P/Z/M)生成鍵盤座標映射"""
        required_keys = ['q', 'p', 'z', 'm']
        if not all(key in anchor_points for key in required_keys):
            print("錨點不足,無法生成鍵盤映射")
            return False

        # 理想鍵盤座標(標準QWERTY佈局)
        src_pts = np.float32([
            [0, 0],      # q
            [9, 0],      # p
            [0.75, 2],   # z
            [6.75, 2]    # m
        ])

        # 實際按鍵位置
        dst_pts = np.float32([
            anchor_points['q'],
            anchor_points['p'],
            anchor_points['z'],
            anchor_points['m']
        ])

        # 計算透視變換矩陣
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 生成所有按鍵的座標
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

        print("四點校準成功,鍵盤映射已優化!")
        return True

    def _generate_finger_map(self):
        """定義每個按鍵應該用哪根手指"""
        return {
            'q': 'RIGHT_PINKY', 'a': 'RIGHT_PINKY', 'z': 'RIGHT_PINKY', 
            'w': 'RIGHT_RING', 's': 'RIGHT_RING', 'x': 'RIGHT_RING',
            'e': 'RIGHT_MIDDLE', 'd': 'RIGHT_MIDDLE', 'c': 'RIGHT_MIDDLE', 
            'r': 'RIGHT_INDEX', 'f': 'RIGHT_INDEX', 'v': 'RIGHT_INDEX',
            't': 'RIGHT_INDEX', 'g': 'RIGHT_INDEX', 'b': 'RIGHT_INDEX', 
            'y': 'LEFT_INDEX', 'h': 'LEFT_INDEX', 'n': 'LEFT_INDEX',
            'u': 'LEFT_INDEX', 'j': 'LEFT_INDEX', 'm': 'LEFT_INDEX', 
            'i': 'LEFT_MIDDLE', 'k': 'LEFT_MIDDLE', 
            'o': 'LEFT_RING', 'l': 'LEFT_RING', 
            'p': 'LEFT_PINKY',
        }

    def draw_keyboard(self, img):
        """在畫面上繪製虛擬鍵盤(只顯示字母)"""
        if not self.key_map: 
            return
        
        for char, (x, y) in self.key_map.items():
            cv2.putText(img, char.upper(), (x - 10, y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def check_fingering(self, typed_key, hands_landmarks, handedness, frame_width, frame_height):
        """檢查輸入的按鍵是否使用正確的手指"""
        if not typed_key or typed_key not in self.key_map: 
            return None, None
        
        # 取得正確的手指
        correct_finger_name = self.finger_map.get(typed_key)
        if not correct_finger_name: 
            return "No rule", None
        
        key_pos = self.key_map[typed_key]
        correct_finger_lm = None
        correct_hand_type = correct_finger_name.split('_')[0]  # RIGHT 或 LEFT
        correct_finger_index = self.fingertip_indices[correct_finger_name]
        
        # 找到正確的手和手指
        for i, hand_lms in enumerate(hands_landmarks):
            hand_type = handedness[i].classification[0].label.upper()
            if hand_type == correct_hand_type:
                correct_finger_lm = hand_lms.landmark[correct_finger_index]
                break
        
        # 檢查正確手指是否在按鍵位置
        if correct_finger_lm:
            dist = ((correct_finger_lm.x * frame_width - key_pos[0])**2 + 
                   (correct_finger_lm.y * frame_height - key_pos[1])**2)**0.5
            if dist < 30: 
                return "Correct", correct_finger_name
        
        # 找出實際使用的手指
        min_dist, actual_finger_name = float('inf'), "Unknown"
        for i, hand_lms in enumerate(hands_landmarks):
            hand_type = handedness[i].classification[0].label.upper()
            for finger, lm_index in self.fingertip_indices.items():
                if finger.startswith(hand_type):
                    lm = hand_lms.landmark[lm_index]
                    dist_actual = ((lm.x * frame_width - key_pos[0])**2 + 
                                  (lm.y * frame_height - key_pos[1])**2)**0.5
                    if dist_actual < min_dist:
                        min_dist, actual_finger_name = dist_actual, finger
        
        if actual_finger_name == correct_finger_name:
            return "Correct", correct_finger_name
            
        return "Wrong", f"Should be {correct_finger_name}, but used {actual_finger_name}"

# 測試代碼(保持註解狀態)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    detector = HandDetector(mode=False, max_hands=2, model_complexity=0, detection_con=0.75, track_con=0.75)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    print("按 ESC 鍵退出測試")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        results = detector.find_hands(image)
        
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        
        cv2.imshow('YiTian - Hand Tracking Test', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
