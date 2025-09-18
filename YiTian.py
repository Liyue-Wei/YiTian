import cv2
import mediapipe as mp
import time
from pynput import keyboard
import threading
import numpy as np
import ttkbootstrap as ttk
import tkinter as tk

def init_screen():
    win = ttk.Window(themename = "darkly")
    win.title("YiTian 盲打輔助器")
    win.resizable(False, False)

    x_COORD = int(win.winfo_screenwidth() / 2 - 320)
    y_COORD = int(win.winfo_screenheight() / 2 - 240)

    win.geometry(f"640x480+{x_COORD}+{y_COORD}")
    win.mainloop()

if __name__ == "__main__":
    init_screen()