# -*- coding: utf-8 -*-
'''
YiTian - Stabilization Module

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

import copy

class Stabilizer:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_landmarks = None

    def process(self, current_landmarks):
        if not current_landmarks:
            self.prev_landmarks = None
            return None

        if self.prev_landmarks is None or len(self.prev_landmarks) != len(current_landmarks):
            self.prev_landmarks = copy.deepcopy(current_landmarks)
            return current_landmarks

        try:
            for i in range(len(current_landmarks)):
                hand = current_landmarks[i]
                prev_hand = self.prev_landmarks[i]

                for j in range(len(hand.landmark)):
                    lm = hand.landmark[j]          
                    prev_lm = prev_hand.landmark[j] 

                    lm.x = self.alpha * lm.x + (1 - self.alpha) * prev_lm.x
                    lm.y = self.alpha * lm.y + (1 - self.alpha) * prev_lm.y
                    lm.z = self.alpha * lm.z + (1 - self.alpha) * prev_lm.z

            self.prev_landmarks = copy.deepcopy(current_landmarks)
        except Exception as e:
            print(f"Error: Stabilization failed: {e}")
        
        return current_landmarks