import cv2
import numpy as np
import time
from multiprocessing import shared_memory
from extmodules import shm_cfg

def main():
    print("正在尝试连接共享内存...")
    
    shm_frame = None
    while shm_frame is None:
        try:
            shm_frame = shared_memory.SharedMemory(name=shm_cfg.SHM_FRAME_ID)
            print(f"成功连接到共享内存: {shm_cfg.SHM_FRAME_ID}")
        except FileNotFoundError:
            print("未找到共享内存，请先运行 YiTian.py... (1秒后重试)")
            time.sleep(1)

    shm_buf = shm_frame.buf
    frame_shape = (shm_cfg.HEIGHT, shm_cfg.WIDTH, shm_cfg.CHANNELS)
    shared_img = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm_buf, offset=1)

    print("开始读取视频流... 将执行 100 秒进行分析...")

    # --- 统计变量初始化 ---
    start_time = time.time()
    duration = 100  # 运行秒数
    count_idle = 0
    count_writing = 0
    # --------------------

    try:
        while True:
            # 检查时间是否到达 100 秒
            elapsed_time = time.time() - start_time
            if elapsed_time > duration:
                print("\n--- 时间到，停止采样 ---")
                break

            # --- 统计逻辑 ---
            # 在做任何判断前，先记录当前状态
            current_status = shm_buf[0]
            
            if current_status == shm_cfg.FLAG_WRITING:
                count_writing += 1
                # 如果正在写入，跳过读取，直接进入下一次循环
                # 注意：这里不需要 sleep，因为我们要尽可能快地采样状态
                continue 
            else:
                count_idle += 1
            # ----------------

            # 4. 读取并显示
            current_frame = shared_img.copy() 
            
            # 打印当前进度
            print(f"\r进度: {elapsed_time:.1f}/{duration}s | Buffer: {hex(id(shm_buf))} | 状态: {current_status}", end="")

            # display_img = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Reader Process", display_img)
            cv2.imshow("Reader Process", current_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户手动停止")
                break

    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # --- 输出分析结果 ---
        total_samples = count_idle + count_writing
        print("\n\n====== 占空比分析报告 ======")
        if total_samples > 0:
            print(f"总采样次数: {total_samples}")
            print(f"IDLE (可读) 次数: {count_idle} \t占比: {count_idle/total_samples*100:.2f}%")
            print(f"WRITING (写入中) 次数: {count_writing} \t占比: {count_writing/total_samples*100:.2f}%")
            
            if count_writing / total_samples > 0.5:
                print("警告：写入状态占比过高，读取进程可能经常被阻塞。")
            else:
                print("状态：读取流畅。")
        else:
            print("未采集到样本。")
        print("============================")
        
        shm_frame.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()