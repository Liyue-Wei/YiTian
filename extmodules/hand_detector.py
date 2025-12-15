# -*- coding: utf-8 -*-
'''
YiTian - Hand Detector Module

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

from multiprocessing import shared_memory
import mediapipe as mp
import numpy as np
import cv2
import time
try:
    import shm_cfg
except ModuleNotFoundError:
    from extmodules import shm_cfg
    
class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=0, detection_con=0.85, track_con=0.85):
        self.shm_frame = None
        self.shm_result = None
        self.float_arr_len = (shm_cfg.RESULT_SIZE - 4) // 4

        try:
            self.shm_frame = shared_memory.SharedMemory(name=shm_cfg.SHM_FRAME_ID)
            print("Process: Shared Memory Frame connected.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Shared Memory unavaliable: {shm_cfg.SHM_FRAME_ID}")
        
        try:
            self.shm_result = shared_memory.SharedMemory(create=True, size=shm_cfg.RESULT_SIZE, name=shm_cfg.SHM_RESULT_ID)
            print("Process: Shared Memory Result created.")
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
            print("Process: Received EXIT Flag.")
            return False
        
        return None
    
    def find_hands(self, img):
        self.result = self.hands.process(img)
        shm_buf = self.shm_result.buf

        if self.result.multi_hand_landmarks:
            count = len(self.result.multi_hand_landmarks)
            shm_buf[0] = min(count, 2)
            result_arr = np.ndarray((self.float_arr_len,), dtype=np.float32, buffer=shm_buf, offset=4)

            index = 0
            for hand_lms, hand_info in zip(self.result.multi_hand_landmarks, self.result.multi_handedness):
                result_arr[index] = 1.0 if hand_info.classification[0].label == 'Left' else 0.0
                result_arr[index+1] = hand_info.classification[0].score
                lms_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                result_arr[index+2 : index+65] = lms_np.flatten()
                index += 65

        else:
            shm_buf[0] = 0

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

def fps_calibration():
    detector = None
    img = None
    elapsed = 0
    try:
        detector = HandDetector()
        while (img := detector.read_img()) is None:
            time.sleep(0.01)

        for _ in range (120):
            while (img := detector.read_img()) is None:
                time.sleep(0.01)
            start = time.perf_counter()
            detector.find_hands(img)
            end = time.perf_counter()
            elapsed += (end - start)
        elapsed = elapsed / 120
        fps = 1 / elapsed if elapsed > 0 else 0

        return fps, elapsed

    except Exception as e:
        return RuntimeError(f"Unexpected error occurred: {e}")
    finally:
        if detector:
            detector.cleanup()

def main():
    detector = None
    try:
        detector = HandDetector()
        while True:
            img = detector.read_img()
            if img is False:
                break
            if img is None:
                continue
            detector.find_hands(img)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if detector:
            detector.cleanup()
        print("Process: Hand Detector closed.")

if __name__ == "__main__":
    main()