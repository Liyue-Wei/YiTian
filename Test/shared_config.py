# 这是一个公共配置文件，Master 和 Slave 都要引用

SHM_FRAME_NAME = "YiTian_Shared_Frame_v1"
SHM_RESULT_NAME = "YiTian_Shared_Result_v1"

# 图像参数
WIDTH = 1280
HEIGHT = 720
CHANNELS = 3
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS # 字节数

# 结果参数 (最多2只手 * 21个点 * 3个坐标(x,y,z) * 4字节float) + 1个字节计数器
# 我们预留大一点的空间，比如 4KB，足够存坐标了
RESULT_SIZE = 4096 

# 状态码 (写入内存的第一个字节)
STATE_IDLE = 0    # 空闲/已读完
STATE_WRITTEN = 1 # Master已写入新图，等待Slave读