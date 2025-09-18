#coding=UTF-8

import cv2
import mediapipe as mp
import time
from pynput import keyboard
import threading
import numpy as np
import ttkbootstrap as ttk
import tkinter as tk

def launch_gui():
    root = ttk.Window(themename="cosmo")
    root.title("YiTian 手語識別系統")
    root.geometry("400x400")
    root.resizable(False, False)

    # 標題
    title = ttk.Label(root, text="YiTian 手語識別系統", font=("Microsoft JhengHei", 18, "bold"))
    title.pack(pady=10)

    # 手指字幕開關
    finger_var = tk.BooleanVar()
    finger_switch = ttk.Checkbutton(root, text="手指字幕", variable=finger_var)
    finger_switch.pack(anchor="w", padx=40, pady=5)

    # 鍵盤字幕開關
    keyboard_var = tk.BooleanVar()
    keyboard_switch = ttk.Checkbutton(root, text="鍵盤字幕", variable=keyboard_var)
    keyboard_switch.pack(anchor="w", padx=40, pady=5)

    # 指引開關
    guide_var = tk.BooleanVar()
    guide_switch = ttk.Checkbutton(root, text="指引開關", variable=guide_var)
    guide_switch.pack(anchor="w", padx=40, pady=5)

    # 攝像頭選擇
    ttk.Label(root, text="攝像頭選擇:").pack(anchor="w", padx=40, pady=(15,0))
    camera_var = tk.StringVar()
    camera_combo = ttk.Combobox(root, textvariable=camera_var, values=["0", "1", "2"], width=10)
    camera_combo.current(0)
    camera_combo.pack(anchor="w", padx=40, pady=5)

    # 保持置頂
    topmost_var = tk.BooleanVar()
    topmost_switch = ttk.Checkbutton(root, text="保持置頂", variable=topmost_var)
    topmost_switch.pack(anchor="w", padx=40, pady=5)

    # 解析度選擇
    ttk.Label(root, text="解析度選擇:").pack(anchor="w", padx=40, pady=(15,0))
    resolution_var = tk.StringVar()
    resolution_combo = ttk.Combobox(root, textvariable=resolution_var, 
                                    values=["640x480", "1280x720", "1920x1080"], width=15)
    resolution_combo.current(0)
    resolution_combo.pack(anchor="w", padx=40, pady=5)

    # 開始按鈕
    def on_start():
        root.destroy()

    start_btn = ttk.Button(root, text="開始", bootstyle="success", command=on_start)
    start_btn.pack(pady=30)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()

