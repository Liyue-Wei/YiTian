import cv2
import mediapipe as mp
import time
from pynput import keyboard
import threading
import numpy as np
import tkinter as tk  

global finger_var, keyboard_var, guide_var, topmost_var, cam_var, res_var

def init_screen():
    win = tk.Tk()
    win.title("YiTian v1.0 Beta")
    win.configure(bg="#0000A0")  
    win.resizable(False, False)

    x_COORD = int(win.winfo_screenwidth() / 2 - 400)
    y_COORD = int(win.winfo_screenheight() / 2 - 300)
    win.geometry(f"800x600+{x_COORD}+{y_COORD}")

    frame = tk.Frame(win, bg="#0000A0", bd=6, relief="ridge", highlightbackground="white", highlightcolor="white", highlightthickness=4)
    frame.place(relx=0.05, rely=0.08, relwidth=0.9, relheight=0.84)

    title_label = tk.Label(frame, text="倚天盲打輔助", font=("DFKai-SB", 28, "bold"), bg="#0000A0", fg="white")
    title_label.grid(row=0, column=0, columnspan=3, pady=30)

    widgets = []

    left_frame = tk.Frame(frame, bg="#0000A0")
    left_frame.grid(row=1, column=0, sticky="nsew", padx=(40, 20), pady=10)

    finger_var = tk.BooleanVar()
    finger_cb = tk.Checkbutton(left_frame, text="手指字幕", variable=finger_var, font=("DFKai-SB", 18), 
                              bg="#0000A0", fg="white", selectcolor="#0000A0", activebackground="#0000A0", 
                              activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", 
                              highlightcolor="white")
    finger_cb.pack(anchor="w", pady=10)
    widgets.append(finger_cb)
    finger_var.set(True)

    keyboard_var = tk.BooleanVar()
    keyboard_cb = tk.Checkbutton(left_frame, text="鍵盤字幕", variable=keyboard_var, font=("DFKai-SB", 18), 
                                bg="#0000A0", fg="white", selectcolor="#0000A0", activebackground="#0000A0", 
                                activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", 
                                highlightcolor="white")
    keyboard_cb.pack(anchor="w", pady=10)
    widgets.append(keyboard_cb)
    keyboard_var.set(True)

    guide_var = tk.BooleanVar()
    guide_cb = tk.Checkbutton(left_frame, text="指引字幕", variable=guide_var, font=("DFKai-SB", 18), 
                             bg="#0000A0", fg="white", selectcolor="#0000A0", activebackground="#0000A0", 
                             activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", 
                             highlightcolor="white")
    guide_cb.pack(anchor="w", pady=10)
    widgets.append(guide_cb)
    guide_var.set(True)

    topmost_var = tk.BooleanVar()
    topmost_cb = tk.Checkbutton(left_frame, text="保持置頂", variable=topmost_var, font=("DFKai-SB", 18), 
                               bg="#0000A0", fg="white", selectcolor="#0000A0", activebackground="#0000A0", 
                               activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", 
                               highlightcolor="white")
    topmost_cb.pack(anchor="w", pady=10)
    widgets.append(topmost_cb)
    topmost_var.set(True)

    right_frame = tk.Frame(frame, bg="#0000A0")
    right_frame.grid(row=1, column=2, sticky="nsew", padx=(20, 40), pady=10)

    cam_label = tk.Label(right_frame, text="攝像頭選擇：", font=("DFKai-SB", 18), bg="#0000A0", fg="white")
    cam_label.pack(anchor="w", pady=(10,0))
    cam_var = tk.StringVar()
    cam_menu = tk.OptionMenu(right_frame, cam_var, "0", "1")
    cam_menu.config(font=("DFKai-SB", 16), bg="#0000A0", fg="white", activebackground="#0000A0", activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", highlightcolor="white")
    cam_menu["menu"].config(bg="#0000A0", fg="white")
    cam_var.set("0")
    cam_menu.pack(anchor="w", pady=10)
    widgets.append(cam_menu)

    res_label = tk.Label(right_frame, text="解析度選擇：", font=("DFKai-SB", 18), bg="#0000A0", fg="white")
    res_label.pack(anchor="w", pady=(10,0))
    res_var = tk.StringVar()
    res_menu = tk.OptionMenu(right_frame, res_var, "640x480", "1280x720", "1920x1080")
    res_menu.config(font=("DFKai-SB", 16), bg="#0000A0", fg="white", activebackground="#0000A0", activeforeground="white", highlightthickness=2, highlightbackground="#0000A0", highlightcolor="white")
    res_menu["menu"].config(bg="#0000A0", fg="white")
    res_var.set("640x480")
    res_menu.pack(anchor="w", pady=10)
    widgets.append(res_menu)

    btn = tk.Button(
        frame, text="開始", font=("DFKai-SB", 20, "bold"),
        bg="#0000A0", fg="white", activebackground="white",
        activeforeground="#0000A0", bd=3, relief="ridge",
        highlightthickness=2, highlightbackground="#0000A0", highlightcolor="white"
    )
    btn.grid(row=2, column=0, columnspan=3, pady=40)
    widgets.append(btn)

    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)
    frame.grid_columnconfigure(2, weight=1)

    def on_focus_in(event):
        widget = event.widget
        widget_type = widget.winfo_class()
        if widget_type == "Checkbutton":
            widget.config(
                highlightbackground="white",
                bg="red",
                fg="white",
                activebackground="red",
                activeforeground="white",
                selectcolor="red"
            )
        elif widget_type == "Button":
            widget.config(
                highlightbackground="white",
                bg="red",
                fg="white",
                activebackground="red",
                activeforeground="white"
            )
        elif widget_type == "Menubutton":
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
        if widget_type == "Checkbutton":
            widget.config(
                highlightbackground="#0000A0",
                bg="#0000A0",
                fg="white",
                activebackground="#0000A0",
                activeforeground="white",
                selectcolor="#0000A0"
            )
        elif widget_type == "Button":
            widget.config(
                highlightbackground="#0000A0",
                bg="#0000A0",
                fg="white",
                activebackground="#0000A0",
                activeforeground="white"
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

    def navigate(event):
        try:
            current_widget = win.focus_get()
            current_index = widgets.index(current_widget)
            
            if event.keysym == 'Down':
                next_index = (current_index + 1) % len(widgets)
            elif event.keysym == 'Up':
                next_index = (current_index - 1 + len(widgets)) % len(widgets)
            else:
                return

            widgets[next_index].focus_set()
        except (ValueError, AttributeError):
            if widgets:
                widgets[0].focus_set()

    win.bind('<Down>', navigate)
    win.bind('<Up>', navigate)

    widgets[0].focus_set()  

    win.mainloop()

def main():
    pass

if __name__ == "__main__":
    init_screen()
    main()