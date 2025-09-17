modules = ["ttkbootstrap", "mediapipe", "opencv-python", "numpy", "pynput", "torch"]
print("preparing to install {}".format(modules))

import os

os.system("python.exe -m pip install --upgrade pip")

for module in modules:
    os.system("pip install "+module)

try:
    os.system('nvidia-smi.exe')
    os.system("pip install "+"cupy-cuda12x")    #根据Cuda版本變更cupy安装版本，參考https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

except:
    print("Cuda environment not avaliable")
    os.system('pause')