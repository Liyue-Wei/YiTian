import time
import struct
import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import shared_memory
import shared_config as cfg

def main():
    print("[Slave] 正在分配共享内存...")
    
    # --- 1. 创建共享内存块 ---
    try:
        # 如果上次非正常退出，内存可能还存在，先尝试连接并释放
        existing_shm = shared_memory.SharedMemory(name=cfg.SHM_FRAME_NAME)
        existing_shm.close()
        existing_shm.unlink()
    except FileNotFoundError:
        pass

    try:
        # 创建新的共享内存
        shm_frame = shared_memory.SharedMemory(name=cfg.SHM_FRAME_NAME, create=True, size=cfg.FRAME_SIZE + 1) # +1 用于状态位
        shm_result = shared_memory.SharedMemory(name=cfg.SHM_RESULT_NAME, create=True, size=cfg.RESULT_SIZE)
    except FileExistsError:
        print("[Error] 共享内存已存在，请手动清理或重启。")
        return

    # 将共享内存映射为 numpy 数组，实现零拷贝访问
    # 注意：buffer[1:] 是图像数据，buffer[0] 是状态位
    frame_buffer = np.ndarray((cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS), dtype=np.uint8, buffer=shm_frame.buf, offset=1)
    
    print(f"[Slave] 内存映射成功。地址: {cfg.SHM_FRAME_NAME}")
    print("[Slave] 初始化 MediaPipe...")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0)

    print("[Slave] 准备就绪，等待数据...")

    try:
        while True:
            # --- 2. 检查状态位 (自旋锁) ---
            # 读取第一个字节
            state = shm_frame.buf[0]
            
            if state == cfg.STATE_WRITTEN:
                # Master 已经写好了新的一帧，我们可以读了
                
                # --- 3. 直接从内存读取图像 (Zero Copy) ---
                # 因为 frame_buffer 已经映射到了内存，这里不需要 copy
                img = frame_buffer 
                
                # 转换颜色 (MediaPipe 需要 RGB)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # --- 4. 推理 ---
                results = hands.process(img_rgb)
                
                # --- 5. 写入结果到共享内存 ---
                # 格式设计: [手数量(int), x1, y1, z1, x2, y2, z2...] 所有都是 float
                data_list = []
                hand_count = 0
                
                if results.multi_hand_landmarks:
                    hand_count = len(results.multi_hand_landmarks)
                    for hand_lms in results.multi_hand_landmarks:
                        for lm in hand_lms.landmark:
                            data_list.extend([lm.x, lm.y, lm.z])
                
                # 打包二进制数据
                # 'I' 是 unsigned int (手数量), 'f'*N 是 N 个 float
                pack_format = 'I' + 'f' * len(data_list)
                packed_data = struct.pack(pack_format, hand_count, *data_list)
                
                # 写入 Result 内存块
                shm_result.buf[:len(packed_data)] = packed_data
                
                # --- 6. 重置状态位 ---
                # 告诉 Master：我算完了，你可以写下一帧了
                shm_frame.buf[0] = cfg.STATE_IDLE
            
            else:
                # 如果没有新数据，稍微休息一下避免死循环占满 CPU
                # 极短的 sleep，几乎不影响延迟
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("[Slave] 停止运行...")
    finally:
        # 清理内存，非常重要！否则下次运行会报错
        shm_frame.close()
        shm_frame.unlink()
        shm_result.close()
        shm_result.unlink()
        print("[Slave] 内存已释放。")

if __name__ == "__main__":
    main()