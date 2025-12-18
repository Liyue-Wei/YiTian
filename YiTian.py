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
import cv2
import subprocess
import numpy as np
import time
# import customtkinter as ctk
# import pywinstyles
# from PIL import Image, ImageTk
import tkinter as tk

def camera(num, stop_event, ready_event):
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
            print("Process: Shared Memory Frame created.")
            ready_event.set()
        except FileExistsError:
            print("Process: Shared Memory Frame already exists. Cleaning up...")
            with shared_memory.SharedMemory(name=shm_cfg.SHM_FRAME_ID) as temp_shm:
                temp_shm.unlink()
            shm_frame = shared_memory.SharedMemory(create=True, size=shm_cfg.FRAME_SIZE, name=shm_cfg.SHM_FRAME_ID)

        shm_buf = shm_frame.buf
        shm_buf[0] = shm_cfg.FLAG_IDLE
        frame_array = np.ndarray((res[1], res[0], shm_cfg.CHANNELS), dtype=np.uint8, buffer=shm_buf, offset=1)

        while not stop_event.is_set():
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
                shm_frame = None
                print("Process: Shared Memory cleared.")
            except Exception as e:
                print(f"Error: Cleaning up shared memory failed: {e}")

class YiTian:
    def __init__(self):
        self.stop_event = multiprocessing.Event()
        self.ready_event = multiprocessing.Event()
        self.cam_proc = None
        self.hd = None
        self.kbl = None
        self.fc = None

    def start_cam(self):
        print("Process: Starting Camera.")

if __name__ == "__main__":
    main = YiTian()