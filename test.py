# ...existing code...
import threading
from pynput import keyboard
import time # 引入 time 用於測試循環

class KeyboardListener:
    def __init__(self):
        self.pressed_key = None
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        with self.lock:
            # 【实验模式】不加 hasattr 检查，直接硬拿 .char
            try:
                self.pressed_key = key.char
            except AttributeError:
                # 如果不加 try-except，线程会直接死掉
                # 这里把它打印出来让你看到报错
                print(f"\n[Error] 抓到了！按键 {key} 居然没有 .char 属性！")
            # print(key)

    def get_last_key(self):
        with self.lock:
            k = self.pressed_key
            self.pressed_key = None
            return k

if __name__ == "__main__":
    l = KeyboardListener()
    print("开始监听... 请尝试按：")
    print("1. 普通键 (a, b, 1) -> 应该正常")
    print("2. 特殊键 (Space, Shift, Enter) -> 应该报错")
    
    while True:
        k = l.get_last_key()
        if k:
            print(f"主程序获取到: {k}")
        time.sleep(0.1)