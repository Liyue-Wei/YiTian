import sys
import subprocess

packages = ["ttkbootstrap", 
            "pillow", 
            "mediapipe", 
            "opencv-contrib-python", 
            "numpy", 
            "pynput"]

def installer(package):
    pass

def main():
    print(f"Currient version = Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    if sys.version_info < (3, 10):
        print(f"Error: Python 3.10 or higher is required. Terminated...")
        sys.exit()
    
if __name__ == "__main__":
    main()