import sys
import subprocess

packages = ["ttkbootstrap", 
            "pillow", 
            "mediapipe", 
            "opencv-contrib-python", 
            "numpy", 
            "pynput"]

def installer(package):
    print(f"[+] Installing {package}")
    try:
        pass
    except:
        pass

def main():
    print(f"Currient version: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    if sys.version_info < (3, 10):
        print(f"Error: Python 3.10 or higher is required. Terminated...")
        sys.exit()

    for package in packages:
        installer(package)
    
if __name__ == "__main__":
    main()