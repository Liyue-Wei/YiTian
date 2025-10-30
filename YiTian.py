import cv2
import mediapipe as mp
from pynput import keyboard
import threading
import copy  

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
                elif key == keyboard.Key.space:
                    self.last_key = ' '
                elif key == keyboard.Key.enter:
                    self.last_key = '\n'
                elif key == keyboard.Key.tab:
                    self.last_key = '\t'
                elif key == keyboard.Key.backspace:
                    self.last_key = 'BACKSPACE'
                elif key == keyboard.Key.esc:
                    self.last_key = 'ESC'
            except AttributeError:
                pass

    def get_last_key(self):
        with self.lock:
            key = self.last_key
            self.last_key = None
            return key
        
class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=0, detection_con=0.75, track_con=0.75):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_con, self.track_con)
        self.results = None
        self.prev_landmarks = None 

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results and self.results.multi_hand_landmarks:
            current_hand_count = len(self.results.multi_hand_landmarks)
            
            if self.prev_landmarks and len(self.prev_landmarks) == current_hand_count:
                alpha = 0.5
                try:
                    for i in range(current_hand_count):
                        hand = self.results.multi_hand_landmarks[i]
                        prev_hand = self.prev_landmarks[i]
                        for j in range(len(hand.landmark)):
                            lm = hand.landmark[j]
                            prev_lm = prev_hand.landmark[j]
                            lm.x = alpha * lm.x + (1 - alpha) * prev_lm.x
                            lm.y = alpha * lm.y + (1 - alpha) * prev_lm.y
                            lm.z = alpha * lm.z + (1 - alpha) * prev_lm.z
                except (IndexError, AttributeError) as e:
                    print(f"平滑處理錯誤: {e}")
                    pass
            
            self.prev_landmarks = copy.deepcopy(self.results.multi_hand_landmarks)
        else:
            self.prev_landmarks = None
        
        return self.results
    
'''
# 測試代碼 - 顯示手部追蹤效果
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = HandDetector()
    
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    print("按 ESC 鍵退出測試")
    print("當前設定: detection_con=0.3, track_con=0.3, alpha=0.7")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # 偵測手部
        results = detector.find_hands(image)
        
        # 繪製手部骨架
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        
        # 顯示參數資訊
        cv2.putText(image, f"Detection: {detector.detection_con} | Track: {detector.track_con}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, "Smooth: alpha=0.7", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('YiTian - Hand Tracking Test', image)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 鍵
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("測試結束")
'''