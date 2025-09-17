#coding=UTF-8
start = False
while (start == False):
    try:
        import cv2
        import mediapipe as mp
        import time
        from pynput import keyboard
        import threading
        import numpy as np
        import ttkbootstrap as ttk
        import tkinter as tk

    except ImportError:
        print("ERROR : Essential modules not found")
        import os
        from Extension_modules import install
        install.main()  
        os.system("PAUSE") 