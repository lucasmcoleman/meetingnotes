#!/usr/bin/env python3
"""
Installation script for MeetingNotes with automatic AMD ROCm GPU detection.
Installs PyTorch with appropriate backend (ROCm for AMD GPUs, default otherwise).
Creates a virtual environment to isolate dependencies.
"""

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


def detect_os():
    """Detect the operating system."""
    system = platform.system()
    if system == "Linux":
        return "linux"
    elif system == "Darwin":
        return "mac"
    elif system == "Windows":
        return "windows"
    else:
        return "unknown"


def get_venv_path():
    """Get the path to the virtual environment directory."""
    return Path(".venv")


def get_venv_pip():
    """Get the path to pip executable in the virtual environment."""
    venv_path = get_venv_path()
    os_type = detect_os()
    
    if os_type == "windows":
        return str(venv_path / "Scripts" / "pip.exe")
    else:
        return str(venv_path / "bin" / "pip")


def get_venv_python():
    """Get the path to python executable in the virtual environment."""
    venv_path = get_venv_path()
    os_type = detect_os()
    
    if os_type == "windows":
        return str(venv_path / "Scripts" / "python.exe")
    else:
        return str(venv_path / "bin" / "python")


def create_venv():
    """Create a virtual environment if it doesn't exist."""
    venv_path = get_venv_path()
    
    if venv_path.exists():
        print(f"✓ Virtual environment already exists at {venv_path}")
        return
    
    print(f"\n🔧 Creating virtual environment at {venv_path}...")
    try:
        venv.create(venv_path, with_pip=True)
        print(f"✓ Virtual environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        sys.exit(1)


def detect_amd_gpu_linux():
    """
    Detect AMD GPU on Linux by checking for ROCm device or AMD GPU in lspci.
    Returns True if AMD GPU is detected, False otherwise.
    """
    # Check for ROCm kernel device
    if os.path.exists("/dev/kfd"):
        print("✓ Detected ROCm kernel device (/dev/kfd)")
        return True
    
    # Check lspci for AMD GPU (vendor ID 1002)
    try:
        result = subprocess.run(
            ["lspci", "-n"],
            capture_output=True,
            text=True,
            check=True
        )
        if "1002:" in result.stdout:
            print("✓ Detected AMD GPU via lspci")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # lspci not available or failed
        pass
    
    return False


def install_pytorch(os_type, has_amd_gpu):
    """
    Install PyTorch with appropriate backend using venv pip.
    
    Args:
        os_type: Operating system type (linux, mac, windows, unknown)
        has_amd_gpu: Boolean indicating if AMD GPU is detected
    """
    pip_executable = get_venv_pip()
    
    if os_type == "linux" and has_amd_gpu:
        # Install PyTorch with ROCm support
        print("\n🔴 Installing PyTorch with AMD ROCm support (ROCm 6.0)...")
        pip_command = [
            pip_executable, "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/rocm6.0"
        ]
    else:
        # Install standard PyTorch (CPU or CUDA if available)
        print("\n📦 Installing PyTorch with default backend...")
        pip_command = [
            pip_executable, "install",
            "torch", "torchvision", "torchaudio"
        ]
    
    try:
        subprocess.run(pip_command, check=True)
        print("✓ PyTorch installation complete")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        sys.exit(1)


def install_requirements():
    """Install remaining requirements from requirements.txt using venv pip."""
    print("\n📦 Installing remaining dependencies from requirements.txt...")
    pip_executable = get_venv_pip()
    
    try:
        subprocess.run(
            [pip_executable, "install", "-r", "requirements.txt"],
            check=True
        )
        print("✓ All dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)


def print_activation_instructions():
    """Print instructions for activating the virtual environment."""
    os_type = detect_os()
    venv_path = get_venv_path()
    
    print("\n" + "=" * 60)
    print("📌 IMPORTANT: Activate the virtual environment")
    print("=" * 60)
    
    if os_type == "windows":
        print(f"\nOn Windows (Command Prompt):")
        print(f"  {venv_path}\\Scripts\\activate.bat")
        print(f"\nOn Windows (PowerShell):")
        print(f"  {venv_path}\\Scripts\\Activate.ps1")
    else:
        print(f"\nOn Linux/macOS:")
        print(f"  source {venv_path}/bin/activate")
    
    print("\nThen run the application with:")
    print("  python main.py")
    print("\nTo deactivate the virtual environment later:")
    print("  deactivate")


def main():
    """Main installation flow."""
    print("=" * 60)
    print("MeetingNotes Installation Script")
    print("=" * 60)
    
    # Detect OS
    os_type = detect_os()
    print(f"\n🖥️  Detected OS: {os_type}")
    
    # Create virtual environment
    create_venv()
    
    # Detect AMD GPU (Linux only)
    has_amd_gpu = False
    if os_type == "linux":
        print("\n🔍 Checking for AMD GPU...")
        has_amd_gpu = detect_amd_gpu_linux()
        if has_amd_gpu:
            print("   → Will install PyTorch with ROCm support")
        else:
            print("   → No AMD GPU detected, using default PyTorch")
    
    # Install PyTorch
    install_pytorch(os_type, has_amd_gpu)
    
    # Install remaining requirements
    install_requirements()
    
    print("\n" + "=" * 60)
    print("✓ Installation complete!")
    print("=" * 60)
    if has_amd_gpu:
        print("\n🔴 AMD ROCm support enabled")
        print("   You can use 'ROCm' mode in the application for GPU acceleration")
    
    # Print activation instructions
    print_activation_instructions()


if __name__ == "__main__":
    main()