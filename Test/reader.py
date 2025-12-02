import cv2
import numpy as np
import time
from multiprocessing import shared_memory
from extmodules import shm_cfg

def main():
    print("正在尝试连接共享内存...")
    
    # 1. 尝试连接共享内存 (如果主程序没启动，这里会报错，所以加个循环等待)
    shm_frame = None
    while shm_frame is None:
        try:
            shm_frame = shared_memory.SharedMemory(name=shm_cfg.SHM_FRAME_ID)
            print(f"成功连接到共享内存: {shm_cfg.SHM_FRAME_ID}")
        except FileNotFoundError:
            print("未找到共享内存，请先运行 YiTian.py... (1秒后重试)")
            time.sleep(1)

    # 2. 映射共享内存中的数组
    # 注意：形状必须与 YiTian.py 中写入的一致 (Height, Width, Channels)
    shm_buf = shm_frame.buf
    frame_shape = (shm_cfg.HEIGHT, shm_cfg.WIDTH, shm_cfg.CHANNELS)
    
    # offset=1 是为了跳过第0个字节的标志位
    shared_img = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm_buf, offset=1)

    print("开始读取视频流... 按 'q' 退出")

    try:
        while True:
            # 3. 检查标志位
            # 如果正在写入 (FLAG_WRITING)，则跳过这一轮读取，防止画面撕裂
            if shm_buf[0] == shm_cfg.FLAG_WRITING:
                continue

            # 4. 读取并显示
            # 共享内存里是 RGB，OpenCV 显示需要 BGR，所以要转回来
            # copy() 是为了把数据从共享内存复制到本地，防止显示过程中数据被主程序修改
            current_frame = shared_img.copy() 
            display_img = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Reader Process", display_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理工作
        shm_frame.close()
        cv2.destroyAllWindows()

main()