import cv2
import mediapipe as mp
import numpy as np
import time
from pynput import keyboard
import tkinter as tk

def init_screen():
    win = tk.Tk()
    win.title("YiTian v1.0 Beta")
    win.configure(bg="#0000A0")  
    win.resizable(False, False)

    x_COORD = int(win.winfo_screenwidth() / 2 - 400)
    y_COORD = int(win.winfo_screenheight() / 2 - 300)
    win.geometry(f"800x600+{x_COORD}+{y_COORD}")

    settings = {}

    def btnPressed():
        settings['finger'] = finger_var.get()
        settings['keyboard'] = keyboard_var.get()
        settings['guide'] = guide_var.get()
        settings['topmost'] = topmost_var.get()
        settings['camera_index'] = int(cam_var.get())
        res_w, res_h = map(int, res_var.get().split('x'))
        settings['resolution'] = (res_w, res_h)
        win.destroy()
    
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
    res_var.set("1280x720")
    res_menu.pack(anchor="w", pady=10)
    widgets.append(res_menu)

    btn = tk.Button(
        frame, text="開始", font=("DFKai-SB", 20, "bold"),
        bg="#0000A0", fg="white", activebackground="white",
        activeforeground="#0000A0", bd=3, relief="ridge",
        highlightthickness=2, highlightbackground="#0000A0", 
        highlightcolor="white", command=btnPressed
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
    return settings


def main(settings):
    print(settings)
    cam_index = settings["camera_index"]
    width, height = settings["resolution"]

    try:
        window_name = "YiTian v1.0 Beta"
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise ConnectionError(f"無法開啟攝影機 {cam_index}")

        print(f"正在嘗試設定解析度為: {width}x{height}...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"攝影機實際解析度為: {actual_width}x{actual_height}")
        if actual_width != width or actual_height != height:
            raise ValueError(f"錯誤：攝影機不支援所選解析度 {width}x{height}。")

        print("攝影機已成功啟動。") 
        start_time = time.time() # 記錄主迴圈開始的時間

        while True:
            success, frame = cap.read()
            if not success:
                break

            if time.time() - start_time < 3:
                text = "Press ESC to exit."
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27: 
                break

            if settings.get('topmost'):
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"開啟攝影機時發生錯誤: {e}, 程式已終止。\n")
        return

if __name__ == "__main__":
    user_settings = init_screen()
    if user_settings:
        main(user_settings)