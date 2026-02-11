#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
import re
import os

# ---------- Configuration ----------
# Each library's source directory (relative to this script)
LIBS = [
    {"name": "sageattention", "path": "SageAttention", "version_file": "sageattention/_version.py"},
    # Add more libraries here if needed
    # {"name": "otherlib", "path": "OtherLib"},
]

# ---------- Helper Functions ----------
def get_expected_version(lib_root: Path, version_file: str) -> str:
    """
    Extract __version__ from _version.py
    """
    version_path = lib_root / version_file
    if not version_path.exists():
        raise FileNotFoundError(f"{version_path} not found")

    content = version_path.read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", content, re.M)
    if not match:
        raise ValueError(f"Cannot extract __version__ from {version_path}")
    return match.group(1)

def get_installed_version(lib_name: str) -> str:
    """Get the installed version of the library; return empty string if not installed"""
    try:
        module = __import__(lib_name)
        return getattr(module, "__version__", "")
    except ModuleNotFoundError:
        return ""

def install_library(lib_path: Path):
    """Run python setup.py install"""
    cmd = [sys.executable, "setup.py", "install"]
    subprocess.check_call(cmd, cwd=str(lib_path))

def ensure_torch():
    """Ensure PyTorch is installed in the current environment"""
    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is not installed. Installing torch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

def is_cuda_sm80() -> bool:
    """Check if current platform is CUDA and SM=80"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            if cap == (8, 0):
                return True
        return False
    except ImportError:
        sm = os.environ.get("CUDA_SM", "")
        return sm == "80"

# ---------- Main Logic ----------
def main():
    if not is_cuda_sm80():
        print("Current platform is not CUDA or not SM=80, skipping installation.")
        return

    # Ensure torch is installed
    ensure_torch()

    script_dir = Path(__file__).parent

    for lib in LIBS:
        lib_path = script_dir / lib["path"]
        lib_name = lib["name"]

        expected_version = get_expected_version(
            lib_path,
            lib["version_file"]
        )

        installed_version = get_installed_version(lib_name)

        print(f"\nChecking library: {lib_name}")
        print(f"Expected version: {expected_version}")
        print(f"Installed version: {installed_version or 'Not installed'}")

        if not installed_version:
            print(f"{lib_name} is not installed. Installing...")
            install_library(lib_path)
        elif installed_version != expected_version:
            print(f"{lib_name} version mismatch. Reinstalling...")
            install_library(lib_path)
        else:
            print(f"{lib_name} is installed and version matches. No action needed.")

if __name__ == "__main__":
    main()
