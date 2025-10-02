"""
Auto-installation script for ComfyUI-Workaround
"""

import subprocess
import sys
import os

def install():
    """Install required packages"""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if os.path.exists(requirements_path):
        print("[ComfyUI-Workaround] Installing dependencies...")
        
        try:
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "-r", 
                requirements_path
            ])
            print("[ComfyUI-Workaround] Installation completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI-Workaround] Installation failed: {e}")
            return False
    else:
        print("[ComfyUI-Workaround] requirements.txt not found")
        return False

if __name__ == "__main__":
    install()