import cv2
import mediapipe as mp
import numpy as np
import time
from pynput import keyboard
import tkinter as tk
import threading

class KeyboardListener:
    def __init__(self):
        self.last_key = None
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        with self.lock:
            try:
                if hasattr(key, 'char') and key.char:
                    self.last_key = key.char.lower()
            except AttributeError:
                pass

    def get_last_key(self):
        with self.lock:
            key = self.last_key
            self.last_key = None
            return key
        
class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=0, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_con, self.track_con)
        self.results = None

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        return self.results
    
class TypingCorrector:
    def __init__(self):
        self.key_layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        self.key_map = {} # 将在校准后动态生成
        self.finger_map = self._generate_finger_map()
        self.fingertip_indices = {
            "RIGHT_THUMB": 4, "RIGHT_INDEX": 8, "RIGHT_MIDDLE": 12, "RIGHT_RING": 16, "RIGHT_PINKY": 20,
            "LEFT_THUMB": 4, "LEFT_INDEX": 8, "LEFT_MIDDLE": 12, "LEFT_RING": 16, "LEFT_PINKY": 20
        }

    def generate_key_map_from_anchors(self, anchor_points):
        if 'q' not in anchor_points or 'p' not in anchor_points:
            print("锚点不足，无法生成键盘映射")
            return False
        q_pos, p_pos = anchor_points['q'], anchor_points['p']
        row_width = p_pos[0] - q_pos[0]
        key_width = row_width / 9
        key_height = key_width * 1.1
        for i, row_str in enumerate(self.key_layout):
            row_offset_x = i * key_width * 0.5
            row_offset_y = i * key_height
            for j, char in enumerate(row_str):
                x = int(q_pos[0] + row_offset_x + j * key_width)
                y = int(q_pos[1] + row_offset_y)
                self.key_map[char] = (x, y)
        print("键盘映射生成成功！")
        return True

    def _generate_finger_map(self):
        return {
            'q': 'RIGHT_PINKY', 'a': 'RIGHT_PINKY', 'z': 'RIGHT_PINKY', 'w': 'RIGHT_RING', 's': 'RIGHT_RING', 'x': 'RIGHT_RING',
            'e': 'RIGHT_MIDDLE', 'd': 'RIGHT_MIDDLE', 'c': 'RIGHT_MIDDLE', 'r': 'RIGHT_INDEX', 'f': 'RIGHT_INDEX', 'v': 'RIGHT_INDEX',
            't': 'RIGHT_INDEX', 'g': 'RIGHT_INDEX', 'b': 'RIGHT_INDEX', 'y': 'LEFT_INDEX', 'h': 'LEFT_INDEX', 'n': 'LEFT_INDEX',
            'u': 'LEFT_INDEX', 'j': 'LEFT_INDEX', 'm': 'LEFT_INDEX', 'i': 'LEFT_MIDDLE', 'k': 'LEFT_MIDDLE', 'o': 'LEFT_RING', 'l': 'LEFT_RING', 'p': 'LEFT_PINKY',
        }

    def draw_keyboard(self, img):
        if not self.key_map: return
        for char, (x, y) in self.key_map.items():
            cv2.putText(img, char.upper(), (x - 18, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

    def check_fingering(self, typed_key, hands_landmarks, handedness, frame_width, frame_height):
        if not typed_key or typed_key not in self.key_map: return None, None
        correct_finger_name = self.finger_map.get(typed_key)
        if not correct_finger_name: return "No rule", None
        key_pos = self.key_map[typed_key]
        correct_finger_lm = None
        correct_hand_type = correct_finger_name.split('_')[0]
        correct_finger_index = self.fingertip_indices[correct_finger_name]
        for i, hand_lms in enumerate(hands_landmarks):
            hand_type = handedness[i].classification[0].label.upper()
            if hand_type == correct_hand_type:
                correct_finger_lm = hand_lms.landmark[correct_finger_index]
                break
        if correct_finger_lm:
            dist = ((correct_finger_lm.x * frame_width - key_pos[0])**2 + (correct_finger_lm.y * frame_height - key_pos[1])**2)**0.5
            if dist < 30: return "Correct", correct_finger_name
        min_dist, actual_finger_name = float('inf'), "Unknown"
        for i, hand_lms in enumerate(hands_landmarks):
            hand_type = handedness[i].classification[0].label.upper()
            for finger, lm_index in self.fingertip_indices.items():
                if finger.startswith(hand_type):
                    lm = hand_lms.landmark[lm_index]
                    dist_actual = ((lm.x * frame_width - key_pos[0])**2 + (lm.y * frame_height - key_pos[1])**2)**0.5
                    if dist_actual < min_dist:
                        min_dist, actual_finger_name = dist_actual, finger
        return "Wrong", f"Should be {correct_finger_name}, but used {actual_finger_name}"

def get_pressing_finger_pos(results, frame_width, frame_height):
    if not results or not results.multi_hand_landmarks: return None
    min_z, finger_pos = float('inf'), None
    for hand_landmarks in results.multi_hand_landmarks:
        for lm_id in [4, 8, 12, 16, 20]:
            lm = hand_landmarks.landmark[lm_id]
            if lm.z < min_z:
                min_z, finger_pos = lm.z, (int(lm.x * frame_width), int(lm.y * frame_height))
    return finger_pos

class TypingTrainerApp:
    def __init__(self, settings):
        self.settings = settings
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.detector = HandDetector(max_hands=2, detection_con=0.5, track_con=0.5, model_complexity=0)
        self.corrector = TypingCorrector()
        self.kb_listener = KeyboardListener()

        # 狀態變數
        self.is_calibrated = False
        self.p_time = 0
        self.last_typed_info = {"key": "", "time": 0}
        self.correction_info = {"msg": "", "details": "", "time": 0}
        
        # 校準相關
        self.calibration_anchors = ['q', 'p']
        self.calibration_points = {}
        self.current_anchor_index = 0

    def _initialize_camera(self):
        """初始化攝影機並設定解析度"""
        camera_index = self.settings.get('camera_index', 0)
        width, height = self.settings.get('resolution', (640, 480))
        
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ConnectionError(f"錯誤：無法開啟攝影機 {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"攝影機實際使用解析度: {self.frame_width} x {self.frame_height}")
        if self.frame_width != width or self.frame_height != height:
            raise ValueError(f"錯誤：攝影機不支援所選解析度 {width}x{height}。")
        return True

    def _run_calibration_phase(self, image, results):
        """處理校準階段的邏輯"""
        target_key = self.calibration_anchors[self.current_anchor_index]
        finger_pos = get_pressing_finger_pos(results, self.frame_width, self.frame_height)
        
        typed_key = self.kb_listener.get_last_key()
        if typed_key == target_key and finger_pos:
            print(f"記錄按鍵 '{target_key}' 位置: {finger_pos}")
            self.calibration_points[target_key] = finger_pos
            self.current_anchor_index += 1

        if self.current_anchor_index >= len(self.calibration_anchors):
            if self.corrector.generate_key_map_from_anchors(self.calibration_points):
                self.is_calibrated = True
                self.last_typed_info["time"] = time.time() # 用於校準完成後的提示
            else:
                self.current_anchor_index = 0
                self.calibration_points = {}

    def _run_practice_phase(self, results):
        """處理練習階段的邏輯"""
        typed_key = self.kb_listener.get_last_key()
        if typed_key:
            self.last_typed_info = {"key": typed_key.upper(), "time": time.time()}
            if results and results.multi_hand_landmarks:
                status, details = self.corrector.check_fingering(typed_key, results.multi_hand_landmarks, results.multi_handedness, self.frame_width, self.frame_height)
                if status:
                    self.correction_info = {"msg": status, "details": details, "time": time.time()}

    def _draw_overlay(self, image, results):
        """在影像上繪製所有UI元素"""
        # 繪製FPS
        c_time = time.time()
        fps = 1 / (c_time - self.p_time) if self.p_time > 0 else 0
        self.p_time = c_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        if not self.is_calibrated:
            # 繪製校準UI
            target_key = self.calibration_anchors[self.current_anchor_index]
            cv2.putText(image, "KEYBOARD CALIBRATION", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
            cv2.putText(image, f"Please press key: '{target_key.upper()}'", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
            finger_pos = get_pressing_finger_pos(results, self.frame_width, self.frame_height)
            if finger_pos:
                cv2.circle(image, finger_pos, 10, (0, 255, 255), 2)
        else:
            # 繪製練習UI
            if time.time() - self.last_typed_info["time"] < 3:
                 cv2.putText(image, "Calibration Complete! Start typing.", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
            
            if self.settings.get('keyboard', False): self.corrector.draw_keyboard(image)
            if self.settings.get('guide', False): cv2.putText(image, "A-Z...", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
            if self.settings.get('finger', False): cv2.putText(image, "LP:QAZ...", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)

            if time.time() - self.last_typed_info["time"] < 2:
                cv2.putText(image, f"Typed: {self.last_typed_info['key']}", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)
            if time.time() - self.correction_info["time"] < 2:
                color = (0, 255, 0) if self.correction_info["msg"] == "Correct" else (0, 0, 255)
                cv2.putText(image, self.correction_info["msg"], (50, 230), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2)
                if self.correction_info["msg"] == "Wrong":
                    cv2.putText(image, self.correction_info["details"], (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def run(self):
        """主應用程式迴圈"""
        try:
            self._initialize_camera()
        except (ConnectionError, ValueError) as e:
            print(e)
            return

        window_name = "Typing Corrector"
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success: break

            results = self.detector.find_hands(image)

            if not self.is_calibrated:
                self._run_calibration_phase(image, results)
            else:
                self._run_practice_phase(results)
            
            self._draw_overlay(image, results)

            cv2.imshow(window_name, image)
            if self.settings.get('topmost', False):
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(1) & 0xFF == 27: break
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.kb_listener.listener.stop()

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
    print("應用程式設定:", settings)
    
    try:
        # 創建並啟動打字訓練應用
        app = TypingTrainerApp(settings)
        
        # 添加初始提示信息的功能
        original_run = app.run
        def run_with_hint():
            print("按ESC鍵退出程式")
            original_run()
        
        app.run = run_with_hint
        app.run()
        
    except Exception as e:
        print(f"應用程式啟動時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    user_settings = init_screen()
    if user_settings:
        main(user_settings)