import os

def main():  
    os.system("python.exe -m pip install --upgrade pip")
    modules = ["ttkbootstrap", "mediapipe", "opencv-python", "opencv-contrib-python", "numpy", "pynput"]  
    print("preparing to install {}".format(modules))
    for i in range(0, len(modules)):
        os.system("pip install "+modules[i])

if __name__ == "__main__":
    main()