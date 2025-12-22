# -*- coding: utf-8 -*-
'''
YiTian - Touch Typing Correction System

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

from extmodules import shm_cfg
from extmodules import keyboard_listener
from extmodules import fingering_corrector
import multiprocessing
from multiprocessing import shared_memory
import sys
import os
import cv2
import subprocess
import numpy as np
import time
# import customtkinter as ctk
# import tkinter as tk

def camera(num, stop_event, ready_event, ui_queue):
    cam = None
    shm_frame = None
    local_key_map = None
    local_finger_pos = None # <--- 新增变量：存储手指位置
    try:
        cam = cv2.VideoCapture(num, cv2.CAP_DSHOW)
        if not cam.isOpened():
            raise IOError(f"Camera {num} can not be opened")
        
        res = [shm_cfg.WIDTH, shm_cfg.HEIGHT, shm_cfg.FPS]
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cam.set(cv2.CAP_PROP_FPS, res[2])
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        real_res = [int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                    int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                    int(cam.get(cv2.CAP_PROP_FPS))]
        
        print(f"Process: Camera opened. Resolution: {real_res}")
        
        try:
            shm_frame = shared_memory.SharedMemory(create=True, size=shm_cfg.FRAME_SIZE, name=shm_cfg.SHM_FRAME_ID)
            print("Process: Shared Memory Frame created.")
            ready_event.set()
        except FileExistsError:
            print("Process: Shared Memory Frame already exists. Cleaning up...")
            try:
                with shared_memory.SharedMemory(name=shm_cfg.SHM_FRAME_ID) as temp_shm:
                    temp_shm.unlink()
            except:
                pass
            shm_frame = shared_memory.SharedMemory(create=True, size=shm_cfg.FRAME_SIZE, name=shm_cfg.SHM_FRAME_ID)
            ready_event.set()

        shm_buf = shm_frame.buf
        shm_buf[0] = shm_cfg.FLAG_IDLE
        frame_array = np.ndarray((res[1], res[0], shm_cfg.CHANNELS), dtype=np.uint8, buffer=shm_buf, offset=1)

        print("Process: Camera loop started.")
        while not stop_event.is_set():
            ret, img = cam.read()
            if not ret:
                continue 
            
            # 1. 检查是否有新数据传过来
            try:
                # 使用 while 循环把队列里的积压数据读完，只保留最新的，防止画面延迟
                while not ui_queue.empty():
                    msg_type, data = ui_queue.get_nowait()
                    if msg_type == 'key_map':
                        local_key_map = data
                        print("Camera: Received Key Map data.")
                    elif msg_type == 'finger_pos': # <--- 新增消息类型
                        local_finger_pos = data
            except:
                pass

            # 2. 准备显示画面
            img_show = img.copy() 

            # 3. 绘制手指当前位置 (红色圆点 + 光圈)
            if local_finger_pos:
                # 画实心红点
                cv2.circle(img_show, local_finger_pos, 6, (0, 0, 255), -1)
                # 画空心圆圈，增加可见度
                cv2.circle(img_show, local_finger_pos, 10, (0, 0, 255), 2)

            # 4. 绘制键盘映射 (如果有)
            if local_key_map:
                for char, (x, y) in local_key_map.items():
                    draw_x = x
                    draw_y = y
                    cv2.circle(img_show, (draw_x, draw_y), 4, (0, 255, 255), -1)
                    cv2.putText(img_show, char.upper(), (draw_x - 10, draw_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("YiTian Camera Feed", img_show)
            cv2.waitKey(1)
            
            try:
                shm_buf[0] = shm_cfg.FLAG_WRITING
                # 注意：写入共享内存的通常是不翻转的原始图像，或者根据 HandDetector 的需求决定
                # 这里我们写入原始图像 img
                frame_array[:] = img
            except:
                pass
            finally:
                shm_buf[0] = shm_cfg.FLAG_IDLE

    except Exception as e:
        print(f"Camera Error: {e}")
    finally:
        # --- 新增：销毁窗口 ---
        cv2.destroyAllWindows()
        # --------------------
        if cam is not None:
            cam.release()
            print("Process: Camera released.")
        
        if shm_frame is not None:
            try:
                shm_buf[0] = shm_cfg.FLAG_EXIT
                print("Process: Sent EXIT flag to HandDector.")
                time.sleep(0.1)
                shm_frame.close()
                shm_frame.unlink()
                print("Process: Shared Memory cleared.")
            except Exception as e:
                print(f"Error: Cleaning up shared memory failed: {e}")

class YiTian:
    def __init__(self):
        self.stop_event = multiprocessing.Event()
        self.ready_event = multiprocessing.Event()
        self.ui_queue = multiprocessing.Queue() # <--- 新增队列
        self.cam_proc = None
        self.hd_proc = None
        self.kbl = None
        self.fc = None

    def start_cam(self, num):
        print("Process: Starting Camera...")
        # 传递 self.ui_queue
        self.cam_proc = multiprocessing.Process(target=camera, args=(num, self.stop_event, self.ready_event, self.ui_queue))
        self.cam_proc.start()

        if not self.ready_event.wait(timeout=15):
            print("Error: Camera initialization timed out.")
            return False
        return True
        
    def start_hd(self):
        print("Process: Starting Hand Detector...")
        try:
            python_exec = sys.executable
            hd_path = os.path.join(os.path.dirname(__file__), "extmodules", "hand_detector.py")
            # 使用 CREATE_NEW_CONSOLE 让它在单独窗口运行，方便调试，发布时可去掉
            self.hd_proc = subprocess.Popen([python_exec, hd_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
            time.sleep(3) # 等待 HD 进程初始化共享内存
            print(f"Process: Hand Detector started with PID: {self.hd_proc.pid}")
            return True
        except Exception as e:
            print(f"Error: Failed to start Hand Detector: {e}")
            return False

    def init_modules(self):
        try:
            print("Process: Initializing Modules...")
            self.kbl = keyboard_listener.KeyboardListener()
            self.fc = fingering_corrector.FingeringCorrector()
            return True
        except Exception as e:
            print(f"Error: Module initialization failed: {e}")
            return False
    
    def _get_calibration_finger(self, hands_data):
        """
        辅助函数：获取用于校准的手指坐标（默认使用检测到的第一只手的食指）
        """
        if not hands_data:
            return None
        # 默认取第一只检测到的手
        hand = hands_data[0]
        # 获取食指指尖 (Index 8)
        return hand['landmarks'][8]

    def run(self):
        # 1. 启动所有子系统
        if not self.start_cam(1): return
        if not self.start_hd(): return
        if not self.init_modules(): return

        print("\n" + "="*40)
        print("YiTian System Started Successfully")
        print("Press 'Ctrl+C' in this terminal to Quit")
        print("="*40 + "\n")

        last_calib_idx = -1 # 用于控制校准提示的打印频率

        # 标记是否已经发送过 key_map，避免重复发送
        key_map_sent = False 

        try:
            while not self.stop_event.is_set():
                # 2. 获取输入数据
                key = self.kbl.get_key()
                hands = self.fc.read_shm_data()

                # --- 新增：发送手指位置给 UI ---
                # 获取用于校准的手指（食指）
                calib_finger = self._get_calibration_finger(hands)
                
                if calib_finger:
                    # 提取 (x, y) 并发送
                    pos_tuple = (calib_finger['x'], calib_finger['y'])
                    self.ui_queue.put(('finger_pos', pos_tuple))
                else:
                    # 如果没检测到手，发送 None 以便 UI 清除光标
                    self.ui_queue.put(('finger_pos', None))

                # 3. 校准模式
                if not self.fc.is_calibrated:
                    # 打印提示信息 (仅当进度变化时)
                    if self.fc.ak_idx != last_calib_idx:
                        target = self.fc.anchor_key[self.fc.ak_idx]
                        print(f">> Calibration: Please press '{target.upper()}' with your finger on the key.")
                        last_calib_idx = self.fc.ak_idx

                    if key:
                        finger_pos = self._get_calibration_finger(hands)
                        if finger_pos:
                            if self.fc.key_map_calibration(key, finger_pos):
                                # 校准成功！发送数据给 Camera 进程
                                print("Main: Sending Key Map to Camera...")
                                self.ui_queue.put(('key_map', self.fc.key_map))
                                key_map_sent = True
                        else:
                            print("Warning: Key pressed but no hand detected!")

                # 4. 纠错模式
                else:
                    # 确保 key_map 被发送过 (防止重启校准后没更新)
                    if not key_map_sent and self.fc.key_map:
                         self.ui_queue.put(('key_map', self.fc.key_map))
                         key_map_sent = True

                    if last_calib_idx != -2:
                        print("\n>> System Ready. Start Typing!\n")
                        last_calib_idx = -2 # 标记为已完成

                    if key:
                        status, detail = self.fc.check_fingering(key, hands)
                        
                        if status == "Correct":
                            print(f"✅ Key: '{key}' | Finger: {detail}")
                        elif status == "Wrong":
                            print(f"❌ Key: '{key}' | Error: Used {detail}")
                        elif status == "Unknown":
                            print(f"⚠️ Key: '{key}' | Hand not detected or too far")
                        # "No rule" (如空格、回车) 忽略不打印

                # 避免 CPU 占用过高
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nUser interrupted.")
        except Exception as e:
            print(f"Runtime Error: {e}")
        finally:
            self.quit()

    def quit(self):
        print("\nProcess: Quitting...")
        self.stop_event.set()
        
        if self.kbl is not None:
            print("Process: Stopping Keyboard Listener.")
            try:
                self.kbl.stop_listener()
            except:
                pass
            self.kbl = None

        if self.cam_proc is not None:
            if self.cam_proc.is_alive():
                self.cam_proc.terminate()
                self.cam_proc.join()
            self.cam_proc = None

        if self.hd_proc is not None:
            self.hd_proc.terminate()
            self.hd_proc = None

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main = YiTian()
    main.run()