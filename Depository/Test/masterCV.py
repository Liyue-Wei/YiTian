import time
import struct
import cv2
import numpy as np
from multiprocessing import shared_memory
import shared_config as cfg

def main():
    # --- 1. 连接共享内存 ---
    try:
        shm_frame = shared_memory.SharedMemory(name=cfg.SHM_FRAME_NAME)
        shm_result = shared_memory.SharedMemory(name=cfg.SHM_RESULT_NAME)
    except FileNotFoundError:
        print("[Error] 找不到共享内存！请先运行 slaveCV.py")
        return

    # 映射 numpy 数组
    frame_buffer = np.ndarray((cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS), dtype=np.uint8, buffer=shm_frame.buf, offset=1)

    print("[Master] 已连接到计算核心。")
    
    cap = cv2.VideoCapture(0)
    cap.set(3, cfg.WIDTH)
    cap.set(4, cfg.HEIGHT)

    latest_hands = [] # 本地缓存

    while True:
        success, img = cap.read()
        if not success: break
        
        # 确保摄像头分辨率匹配，否则内存溢出
        if img.shape != (cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS):
            img = cv2.resize(img, (cfg.WIDTH, cfg.HEIGHT))

        # --- 2. 检查 Slave 是否空闲 ---
        state = shm_frame.buf[0]
        
        if state == cfg.STATE_IDLE:
            # Slave 处于空闲状态，我们可以写入新的一帧
            
            # --- 3. 写入共享内存 (极快) ---
            # 直接将 numpy 数据 copy 到共享内存区
            frame_buffer[:] = img[:] 
            
            # 设置标志位，通知 Slave 开工
            shm_frame.buf[0] = cfg.STATE_WRITTEN
            
            # --- 4. 读取上一帧的计算结果 ---
            # 从 Result 内存块读取
            # 先读前4个字节，获取手的数量
            hand_count = struct.unpack('I', shm_result.buf[:4])[0]
            
            latest_hands = []
            if hand_count > 0:
                # 每个点 3个float(xyz) * 4bytes = 12 bytes
                # 每只手 21个点 = 252 bytes
                # 读取剩余数据
                float_count = hand_count * 21 * 3
                data_bytes = shm_result.buf[4 : 4 + float_count * 4]
                floats = struct.unpack('f' * float_count, data_bytes)
                
                # 重组数据结构
                for i in range(hand_count):
                    hand_points = []
                    base_idx = i * 21 * 3
                    for j in range(21):
                        idx = base_idx + j * 3
                        hand_points.append({
                            'x': floats[idx],
                            'y': floats[idx+1],
                            'z': floats[idx+2]
                        })
                    latest_hands.append(hand_points)

        # --- 5. 绘制 UI (使用最新可用的数据) ---
        cv2.putText(img, f"Shared Memory Mode", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if latest_hands:
            for hand in latest_hands:
                for pt in hand:
                    cx, cy = int(pt['x'] * cfg.WIDTH), int(pt['y'] * cfg.HEIGHT)
                    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)

        cv2.imshow("YiTian - Master (SHM)", img)
        if cv2.waitKey(1) & 0xFF == 27: break

    # 清理
    shm_frame.close()
    shm_result.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()