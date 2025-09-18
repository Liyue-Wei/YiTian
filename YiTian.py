import cv2
import mediapipe as mp
import time
from pynput import keyboard
import threading
import numpy as np
import ttkbootstrap as ttk
import tkinter as tk

def init_screen():
    win = tk.Tk()
    win.title("YiTian v0.1 Beta")
    win.configure(bg="#0000A0")  
    win.resizable(False, False)

    x_COORD = int(win.winfo_screenwidth() / 2 - 320)
    y_COORD = int(win.winfo_screenheight() / 2 - 240)
    win.geometry(f"640x480+{x_COORD}+{y_COORD}")

    frame = tk.Frame(win, bg="#0000A0", bd=6, relief="ridge", highlightbackground="white", highlightcolor="white", highlightthickness=4)
    frame.place(relx=0.05, rely=0.08, relwidth=0.9, relheight=0.84)

    title_label = tk.Label(frame, text="倚天盲打輔助 v0.1 Beta", font=("DFKai-SB", 22, "bold"), bg="#0000A0", fg="white")
    title_label.pack(pady=20)

    # 控件列表，用于Tab切換
    widgets = []

    # 手指字幕開關
    finger_var = tk.BooleanVar()
    finger_cb = tk.Checkbutton(frame, text="手指字幕", variable=finger_var, font=("DFKai-SB", 14), bg="#0000A0", fg="white", selectcolor="#0000A0", activebackground="#0000A0", activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", highlightcolor="white")
    finger_cb.pack(anchor="w", padx=40, pady=2)
    widgets.append(finger_cb)

    # 鍵盤字幕開關
    keyboard_var = tk.BooleanVar()
    keyboard_cb = tk.Checkbutton(frame, text="鍵盤字幕", variable=keyboard_var, font=("DFKai-SB", 14), bg="#0000A0", fg="white", selectcolor="#0000A0", activebackground="#0000A0", activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", highlightcolor="white")
    keyboard_cb.pack(anchor="w", padx=40, pady=2)
    widgets.append(keyboard_cb)

    # 指引開關
    guide_var = tk.BooleanVar()
    guide_cb = tk.Checkbutton(frame, text="指引開關", variable=guide_var, font=("DFKai-SB", 14), bg="#0000A0", fg="white", selectcolor="#0000A0", activebackground="#0000A0", activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", highlightcolor="white")
    guide_cb.pack(anchor="w", padx=40, pady=2)
    widgets.append(guide_cb)

    # 保持置頂
    topmost_var = tk.BooleanVar()
    topmost_cb = tk.Checkbutton(frame, text="保持置頂", variable=topmost_var, font=("DFKai-SB", 14), bg="#0000A0", fg="white", selectcolor="#0000A0", activebackground="#0000A0", activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", highlightcolor="white")
    topmost_cb.pack(anchor="w", padx=40, pady=2)
    widgets.append(topmost_cb)

    # 攝像頭選擇
    cam_label = tk.Label(frame, text="攝像頭選擇：", font=("DFKai-SB", 14), bg="#0000A0", fg="white")
    cam_label.pack(anchor="w", padx=40, pady=(10,0))
    cam_var = tk.StringVar()
    cam_menu = tk.OptionMenu(frame, cam_var, "0", "1")
    cam_menu.config(font=("DFKai-SB", 12), bg="#0000A0", fg="white", activebackground="#0000A0", activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", highlightcolor="white")
    cam_menu["menu"].config(bg="#0000A0", fg="white")
    cam_var.set("0")
    cam_menu.pack(anchor="w", padx=40, pady=2)
    widgets.append(cam_menu)

    # 解析度選擇
    res_label = tk.Label(frame, text="解析度選擇：", font=("DFKai-SB", 14), bg="#0000A0", fg="white")
    res_label.pack(anchor="w", padx=40, pady=(10,0))
    res_var = tk.StringVar()
    res_menu = tk.OptionMenu(frame, res_var, "640x480", "1280x720", "1920x1080")
    res_menu.config(font=("DFKai-SB", 12), bg="#0000A0", fg="white", activebackground="#0000A0", activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", highlightcolor="white")
    res_menu["menu"].config(bg="#0000A0", fg="white")
    res_var.set("640x480")
    res_menu.pack(anchor="w", padx=40, pady=2)
    widgets.append(res_menu)

    # 開始按鈕
    btn = tk.Button(frame, text="開始", font=("DFKai-SB", 16, "bold"), bg="#0000A0", fg="white", activebackground="white", activeforeground="#0000A0", bd=3, relief="ridge", highlightthickness=2, highlightbackground="#0000A0", highlightcolor="white")
    btn.pack(pady=20)
    widgets.append(btn)

    # DOS風格Tab切換高亮（反紅）
    def on_focus_in(event):
        widget = event.widget
        widget_type = widget.winfo_class()
        if widget_type in ["Checkbutton", "Button"]:
            widget.config(
                highlightbackground="white",
                bg="red",
                fg="white",
                activebackground="red",
                activeforeground="white",
                selectcolor="red"
            )
        elif widget_type == "Menubutton":  # OptionMenu的主按钮
            widget.config(
                highlightbackground="white",
                bg="red",
                fg="white",
                activebackground="red",
                activeforeground="white"
            )

    def on_focus_out(event):
        widget = event.widget
        widget_type = widget.winfo_class()
        if widget_type in ["Checkbutton", "Button"]:
            widget.config(
                highlightbackground="#0000A0",
                bg="#0000A0",
                fg="white",
                activebackground="#0000A0",
                activeforeground="white",
                selectcolor="#0000A0"
            )
        elif widget_type == "Menubutton":
            widget.config(
                highlightbackground="#0000A0",
                bg="#0000A0",
                fg="white",
                activebackground="#0000A0",
                activeforeground="white"
            )

    for w in widgets:
        w.bind("<FocusIn>", on_focus_in)
        w.bind("<FocusOut>", on_focus_out)
        w.configure(takefocus=True)

    widgets[0].focus_set()  # 默认聚焦第一个控件

    win.mainloop()

if __name__ == "__main__":
    init_screen()