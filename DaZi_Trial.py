import cv2
import mediapipe as mp
import time
from pynput import keyboard
import threading
import numpy as np

# --- 1. 全局键盘监听器 (无变化) ---
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

# --- 2. 手部检测器 (无变化) ---
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

# --- 3. 指法纠正器 (有修改) ---
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
        """根据'q'和'p'两个锚点生成整个键盘映射"""
        if 'q' not in anchor_points or 'p' not in anchor_points:
            print("锚点不足，无法生成键盘映射")
            return False

        q_pos = anchor_points['q']
        p_pos = anchor_points['p']

        # 假设第一行 ('qwertyuiop') 是线性的
        row_width = p_pos[0] - q_pos[0]
        key_width = row_width / 9  # 'q'到'p'之间有9个间隔
        
        # 假设行高与键宽相似
        key_height = key_width * 1.1 

        for i, row_str in enumerate(self.key_layout):
            # 每一行相对于第一行的偏移量（经验值）
            row_offset_x = i * key_width * 0.5 
            row_offset_y = i * key_height
            
            for j, char in enumerate(row_str):
                x = int(q_pos[0] + row_offset_x + j * key_width)
                y = int(q_pos[1] + row_offset_y)
                self.key_map[char] = (x, y)
        
        print("键盘映射生成成功！")
        return True

    def _generate_finger_map(self):
        # 此部分无变化
        return {
            'q': 'RIGHT_PINKY', 'a': 'RIGHT_PINKY', 'z': 'RIGHT_PINKY',
            'w': 'RIGHT_RING', 's': 'RIGHT_RING', 'x': 'RIGHT_RING',
            'e': 'RIGHT_MIDDLE', 'd': 'RIGHT_MIDDLE', 'c': 'RIGHT_MIDDLE',
            'r': 'RIGHT_INDEX', 'f': 'RIGHT_INDEX', 'v': 'RIGHT_INDEX',
            't': 'RIGHT_INDEX', 'g': 'RIGHT_INDEX', 'b': 'RIGHT_INDEX',
            'y': 'LEFT_INDEX', 'h': 'LEFT_INDEX', 'n': 'LEFT_INDEX',
            'u': 'LEFT_INDEX', 'j': 'LEFT_INDEX', 'm': 'LEFT_INDEX',
            'i': 'LEFT_MIDDLE', 'k': 'LEFT_MIDDLE',
            'o': 'LEFT_RING', 'l': 'LEFT_RING',
            'p': 'LEFT_PINKY',
        }

    def draw_keyboard(self, img):
        if not self.key_map: return
        for char, (x, y) in self.key_map.items():
            cv2.putText(img, char.upper(), (x - 18, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

    def check_fingering(self, typed_key, hands_landmarks, handedness, frame_width, frame_height):
        # 此方法逻辑无变化
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
            DISTANCE_THRESHOLD = 30
            if dist < DISTANCE_THRESHOLD:
                return "Correct", correct_finger_name
        min_dist = float('inf')
        actual_finger_name = "Unknown"
        for i, hand_lms in enumerate(hands_landmarks):
            hand_type = handedness[i].classification[0].label.upper()
            for finger, lm_index in self.fingertip_indices.items():
                if finger.startswith(hand_type):
                    lm = hand_lms.landmark[lm_index]
                    dist_actual = ((lm.x * frame_width - key_pos[0])**2 + (lm.y * frame_height - key_pos[1])**2)**0.5
                    if dist_actual < min_dist:
                        min_dist = dist_actual
                        actual_finger_name = finger
        return "Wrong", f"Should be {correct_finger_name}, but used {actual_finger_name}"

def get_pressing_finger_pos(results, frame_width, frame_height):
    """启发式方法：找到离摄像头最近（Z最小）的指尖作为按键手指"""
    if not results or not results.multi_hand_landmarks:
        return None
    
    min_z = float('inf')
    finger_pos = None
    fingertip_indices = [4, 8, 12, 16, 20]

    for hand_landmarks in results.multi_hand_landmarks:
        for lm_id in fingertip_indices:
            lm = hand_landmarks.landmark[lm_id]
            if lm.z < min_z:
                min_z = lm.z
                finger_pos = (int(lm.x * frame_width), int(lm.y * frame_height))
    return finger_pos

def main():
    p_time = 0
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头实际使用分辨率: {frame_width} x {frame_height}")

    detector = HandDetector(max_hands=2, detection_con=0.5, track_con=0.5, model_complexity=0)
    corrector = TypingCorrector()
    kb_listener = KeyboardListener()

    # --- 校准阶段变量 ---
    is_calibrated = False
    calibration_anchors = ['q', 'p'] # 我们需要定位的锚点按键
    calibration_points = {}
    current_anchor_index = 0

    last_typed_info = {"key": "", "time": 0}
    correction_info = {"msg": "", "details": "", "time": 0}

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        # image = cv2.flip(image, 1)
        results = detector.find_hands(image)

        # ====== 只在校准阶段显示手指缩写与左右手 ======
        if not is_calibrated:
            if results and results.multi_hand_landmarks and results.multi_handedness:
                finger_abbr = ['T', 'I', 'M', 'R', 'P']
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    raw_label = results.multi_handedness[idx].classification[0].label
                    hand_label = 'Right' if raw_label == 'Left' else 'Left'
                    hand_color = (0, 255, 0) if hand_label == 'Left' else (0, 128, 255)
                    wrist = hand_landmarks.landmark[0]
                    wx, wy = int(wrist.x * frame_width), int(wrist.y * frame_height)
                    cv2.putText(image, hand_label, (wx-30, wy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)
                    for i, lm_idx in enumerate([4, 8, 12, 16, 20]):
                        lm = hand_landmarks.landmark[lm_idx]
                        x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                        abbr = ('L' if hand_label == 'Left' else 'R') + finger_abbr[i]
                        cv2.putText(image, abbr, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
        # 根据是否校准，执行不同逻辑
        if not is_calibrated:
            # --- 校准阶段 ---
            target_key = calibration_anchors[current_anchor_index]
            cv2.putText(image, f"KEYBOARD CALIBRATION", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
            cv2.putText(image, f"Please press key: '{target_key.upper()}'", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)

            # 实时检测并显示当前认为是“按键”手指的位置
            finger_pos = get_pressing_finger_pos(results, frame_width, frame_height)
            if finger_pos:
                # 在手指位置画一个圈以方便观察
                cv2.circle(image, finger_pos, 10, (0, 255, 255), 2)
                # 显示实时坐标
                cv2.putText(image, f"Pos: {finger_pos}", (finger_pos[0] + 15, finger_pos[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            typed_key = kb_listener.get_last_key()
            if typed_key == target_key:
                # finger_pos 已在上面获取，这里直接使用
                if finger_pos:
                    print(f"记录按键 '{target_key}' 位置: {finger_pos}")
                    calibration_points[target_key] = finger_pos
                    current_anchor_index += 1
                else:
                    print("未检测到手指，请重试")

            # 检查校准是否完成
            if current_anchor_index >= len(calibration_anchors):
                if corrector.generate_key_map_from_anchors(calibration_points):
                    is_calibrated = True
                else: # 如果生成失败，重置校准
                    current_anchor_index = 0
                    calibration_points = {}
        else:
            # --- 练习阶段 ---
            if time.time() - list(last_typed_info.values())[-1] < 3: # 校准完成后提示3秒
                 cv2.putText(image, "Calibration Complete! Start typing.", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)

            # A~Z 指引教学
            cv2.putText(image, "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
            # 手指指引
            cv2.putText(image, "LP:QAZ  LR:WSX  LM:EDC  LI:RFVTGB  RI:YHNUJM  RM:IK  RR:OL  RP:P", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)

            corrector.draw_keyboard(image) # 显示动态生成的键盘
            typed_key = kb_listener.get_last_key()
            if typed_key:
                last_typed_info = {"key": typed_key.upper(), "time": time.time()}
                if results and results.multi_hand_landmarks:
                    status, details = corrector.check_fingering(typed_key, results.multi_hand_landmarks, results.multi_handedness, frame_width, frame_height)
                    if status:
                        correction_info = {"msg": status, "details": details, "time": time.time()}

            if time.time() - last_typed_info["time"] < 2:
                cv2.putText(image, f"Typed: {last_typed_info['key']}", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)
            if time.time() - correction_info["time"] < 2:
                color = (0, 255, 0) if correction_info["msg"] == "Correct" else (0, 0, 255)
                cv2.putText(image, correction_info["msg"], (50, 230), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2)
                if correction_info["msg"] == "Wrong":
                    cv2.putText(image, correction_info["details"], (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 显示FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if p_time > 0 else 0
        p_time = c_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow('Typing Corrector', image)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()
    kb_listener.listener.stop()

if __name__ == "__main__":
    main()