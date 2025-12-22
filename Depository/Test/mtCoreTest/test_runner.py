import subprocess
import sys
import multiprocessing
import time

def main():
    # 获取 CPU 核心数 (逻辑核心)
    cpu_count = multiprocessing.cpu_count()
    print(f"检测到 {cpu_count} 个 CPU 核心。")
    print("正在启动子进程以吃满所有核心...")
    print("请立即打开【任务管理器】->【性能】选项卡查看 CPU 使用率！")

    processes = []
    
    # 为每个核心启动一个独立的 Python 进程
    for i in range(cpu_count):
        # 使用 sys.executable 确保使用当前的 Python 环境
        p = subprocess.Popen([sys.executable, "./Test/mtCoreTest/cpu_task.py"])
        processes.append(p)
        print(f"已启动进程 {i+1}/{cpu_count}")

    # 等待所有进程结束
    for p in processes:
        p.wait()

    print("所有测试进程已结束。")

if __name__ == "__main__":
    main()