import time
import os

def eat_cpu():
    pid = os.getpid()
    print(f"Process {pid}: 开始全速运算...")
    
    start_time = time.time()
    # 运行 10 秒钟的密集计算
    while time.time() - start_time < 10:
        # 复杂的数学运算，强制 CPU 工作
        _ = [x**2 for x in range(10000)]
        
    print(f"Process {pid}: 任务完成。")

if __name__ == "__main__":
    eat_cpu()