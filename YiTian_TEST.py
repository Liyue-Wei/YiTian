# -*- coding: utf-8 -*-
'''
YiTian - Touch Typing Correction System

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

from extmodules import shm_cfg
from extmodules import keyboard_listener
from extmodules import hand_detector  # 【新增】导入 hand_detector 模块
import multiprocessing
from multiprocessing import shared_memory
import cv2
import subprocess
import numpy as np
import time
# import customtkinter as ctk
# import pywinstyles
# from PIL import Image, ImageTk
import tkinter as tk

# 【修改】增加 sleep_ref 参数，用于动态控制帧率
def camera(num, stop_event, sleep_ref):
    cam = None
    shm_frame = None
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
        if real_res != res:
            raise ValueError(f"Resolution Setting Unavailable: Expected {res}, but got {real_res}")
        else:
            print(f"Process: Camera opened.")
        
        try:
            shm_frame = shared_memory.SharedMemory(create=True, size=shm_cfg.FRAME_SIZE, name=shm_cfg.SHM_FRAME_ID)
            print("Process: Shared Memory created.")
        except FileExistsError:
            print("Process: Shared Memory already exists. Cleaning up...")
            with shared_memory.SharedMemory(name=shm_cfg.SHM_FRAME_ID) as temp_shm:
                temp_shm.unlink()
            shm_frame = shared_memory.SharedMemory(create=True, size=shm_cfg.FRAME_SIZE, name=shm_cfg.SHM_FRAME_ID)

        shm_buf = shm_frame.buf
        shm_buf[0] = shm_cfg.FLAG_IDLE
        frame_array = np.ndarray((res[1], res[0], shm_cfg.CHANNELS), dtype=np.uint8, buffer=shm_buf, offset=1)

        while not stop_event.is_set():
            loop_start = time.perf_counter()  # 【新增】开始计时

            ret, img = cam.read()
            if not ret:
                raise IOError("Frame can not be read")
            
            try:
                shm_buf[0] = shm_cfg.FLAG_WRITING
                frame_array[:] = img
            except:
                pass
            finally:
                shm_buf[0] = shm_cfg.FLAG_IDLE
            
            # 【新增】智能休眠逻辑
            # 计算工作耗时，用目标间隔减去耗时，得到需要 sleep 的时间
            work_cost = time.perf_counter() - loop_start
            target_interval = sleep_ref[0]
            sleep_time = target_interval - work_cost
            
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"Error: {e}")
    finally:
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

# def hand_detector():
#     pass

# def fingering_corrector():
#     pass

# 【新增】绘图工具函数
def draw_landmarks(img, landmarks):
    h, w, _ = img.shape
    # 手指连接关系
    connections = [
        (0,1), (1,2), (2,3), (3,4),         # 拇指
        (0,5), (5,6), (6,7), (7,8),         # 食指
        (0,9), (9,10), (10,11), (11,12),    # 中指
        (0,13), (13,14), (14,15), (15,16),  # 无名指
        (0,17), (17,18), (18,19), (19,20)   # 小指
    ]
    
    # 画点
    for pt in landmarks:
        x, y = int(pt[0] * w), int(pt[1] * h)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    
    # 画线
    for start_idx, end_idx in connections:
        pt1 = landmarks[start_idx]
        pt2 = landmarks[end_idx]
        x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
        x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

def main():
    import threading
    
    # 1. 启动摄像头线程 (生产者)
    stop_event = multiprocessing.Event()
    # 【修改】传入 sleep_ref，初始值为 0 (全速运行)
    sleep_ref = [0.0] 
    cam_thread = threading.Thread(target=camera, args=(1, stop_event, sleep_ref))
    cam_thread.start()
    
    print("Main: Camera thread started.")
    
    # 2. 连接共享内存 (消费者)
    shm_frame_read = None
    shm_result_read = None
    
    try:
        # 等待 camera 创建 SHM_FRAME
        print("Main: Waiting for SHM_FRAME...")
        while not stop_event.is_set():
            try:
                shm_frame_read = shared_memory.SharedMemory(name=shm_cfg.SHM_FRAME_ID)
                print("Main: Connected to SHM_FRAME.")
                break
            except FileNotFoundError:
                time.sleep(0.5)
        
        # 【新增】执行 FPS 校准
        print("Main: Performing FPS Calibration (Please wait)...")
        try:
            # 调用 hand_detector 中的校准函数
            # 注意：这会临时创建一个 HandDetector 实例读取 SHM_FRAME
            calib_fps, calib_elapsed = hand_detector.fps_calibration()
            
            print(f"Main: Calibration Result -> Max FPS: {calib_fps:.2f}, Latency: {calib_elapsed*1000:.2f}ms")
            
            # 设置安全间隔 (1.1倍延迟，留出10%余量)
            safe_interval = calib_elapsed * 1.1
            sleep_ref[0] = safe_interval
            print(f"Main: Camera sync interval set to {safe_interval*1000:.2f}ms (Target FPS: {1/safe_interval:.2f})")
            
        except Exception as e:
            print(f"Main: Calibration failed: {e}. Camera will run at max speed.")

        # 等待 hand_detector 创建 SHM_RESULT (你需要手动运行 hand_detector.py)
        print("Main: Waiting for SHM_RESULT (Please run hand_detector.py)...")
        while not stop_event.is_set():
            try:
                shm_result_read = shared_memory.SharedMemory(name=shm_cfg.SHM_RESULT_ID)
                print("Main: Connected to SHM_RESULT.")
                break
            except FileNotFoundError:
                time.sleep(0.5)

        # 准备读取视图
        frame_shape = (shm_cfg.HEIGHT, shm_cfg.WIDTH, shm_cfg.CHANNELS)
        result_float_len = (shm_cfg.RESULT_SIZE - 4) // 4
        
        print("Main: Starting visualization loop...")
        while not stop_event.is_set():
            # --- A. 读取图片 ---
            # 注意：这里我们作为"观察者"读取，不修改标志位，只读数据
            # 为了防止读到写了一半的数据，可以简单判断一下 FLAG
            if shm_frame_read.buf[0] == shm_cfg.FLAG_WRITING:
                continue # 正在写，跳过这一帧
                
            # 零拷贝读取
            frame_arr = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm_frame_read.buf, offset=1)
            display_img = frame_arr.copy() # 拷贝一份出来画图，不影响原图
            
            # --- B. 读取结果并绘制 ---
            try:
                num_hands = shm_result_read.buf[0]
                if num_hands > 0:
                    res_arr = np.ndarray((result_float_len,), dtype=np.float32, buffer=shm_result_read.buf, offset=4)
                    
                    idx = 0
                    for i in range(num_hands):
                        # 提取坐标 (跳过 Label 和 Score)
                        landmarks = res_arr[idx+2 : idx+65].reshape(21, 3)
                        draw_landmarks(display_img, landmarks)
                        
                        # 显示置信度
                        score = res_arr[idx+1]
                        label = "Right" if res_arr[idx] > 0.5 else "Left"
                        cv2.putText(display_img, f"{label}: {score:.2f}", (10, 30 + i*30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        idx += 65
            except Exception as e:
                pass # 偶尔读写冲突忽略即可

            # --- C. 显示 ---
            cv2.imshow("YiTian Visualization", display_img)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("Main: Stopping...")
        stop_event.set()
        cam_thread.join()
        
        if shm_frame_read:
            shm_frame_read.close()
        if shm_result_read:
            shm_result_read.close()
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()