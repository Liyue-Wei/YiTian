# -*- coding: utf-8 -*-
'''
YiTian - Keyboard Listener Module

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

import os
import sys
import subprocess

packages = ["ttkbootstrap", 
            "pillow", 
            "mediapipe", 
            "opencv-contrib-python", 
            "numpy", 
            "pynput"]

success = []
exists = []
not_found = []
failed = []

def installer(package):
    print(f"[*] Processing {package}...") 
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package], 
        capture_output=True, 
        text=True
    )

    if result.returncode == 0:
        if "Requirement already satisfied" in result.stdout:
            print(f"  [=] Already installed.")
            exists.append(package)
        else:
            print(f"  [+] Successfully installed.")
            success.append(package)
            
    else:
        if "No matching distribution found" in result.stderr:
            print(f"  [-] Package not found in PyPI.")
            not_found.append(package)
        else:
            print(f"  [!] Installation failed (Network/Permission error).")
            failed.append(package)

def main():
    print(f"Current version: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    if sys.version_info < (3, 10):
        print(f"Error: Python 3.10 or higher is required. Terminated...")
        sys.exit()

    for package in packages:
        installer(package)

    print("\n" + "="*36)
    print("        INSTALLATION SUMMARY        ")
    print("="*36)
    
    if success:
        print(f"[+] Installed ({len(success)}): {', '.join(success)}")
    if exists:
        print(f"[=] Already Exists ({len(exists)}): {', '.join(exists)}")
    if not_found:
        print(f"[-] Not Found ({len(not_found)}): {', '.join(not_found)}")
    if failed:
        print(f"[!] Failed ({len(failed)}): {', '.join(failed)}")
    
    print()
    os.system("PAUSE")

if __name__ == "__main__":
    main()