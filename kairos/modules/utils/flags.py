import os
import torch
import subprocess

def detectplat_device_gpu_vendor():
    try:
        result = subprocess.run(
            ["mx-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
        output = result.stdout + result.stderr

        if "MetaX" in output:
            return "MetaX"
    except:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
        if "NVIDIA" in result.stdout:
            return "NVIDIA"
    except:
        pass

    return "Unknown"


def get_cuda_sm():
    IS_CUDA = torch.cuda.is_available()
    if not IS_CUDA:
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor 


FLAGS_KAIROS_PLAT_DEVICE = detectplat_device_gpu_vendor()
FLAGS_KAIROS_IS_METAX = False
FLAGS_KAIROS_CUDA_SM = None
print(f'current device: {FLAGS_KAIROS_PLAT_DEVICE}')

if FLAGS_KAIROS_PLAT_DEVICE == "NVIDIA":
    FLAGS_KAIROS_CUDA_SM = get_cuda_sm()

if FLAGS_KAIROS_PLAT_DEVICE == "MetaX":
    os.environ["IS_METAX"] = "1" # for docker 
    os.environ["FLAGS_KAIROS_IS_METAX"] = "1"
    FLAGS_KAIROS_IS_METAX = True
    if not hasattr(torch, "maca"):
        torch.maca = torch.cuda

    if hasattr(torch, "get_autocast_dtype"):
        _orig_get_autocast_dtype = torch.get_autocast_dtype

        def _patched_get_autocast_dtype(device_type: str):
            # fla / triton ~G~L~B~^~\| ~F 'maca'~L~_~@~S~H~P 'cuda'
            if device_type == "maca":
                device_type = "cuda"
            return _orig_get_autocast_dtype(device_type)

        torch.get_autocast_dtype = _patched_get_autocast_dtype

    if hasattr(torch, "is_autocast_enabled"):
        _orig_is_autocast_enabled = torch.is_autocast_enabled

        def _patched_is_autocast_enabled(device_type: str = "cuda"):
            if device_type == "maca":
                device_type = "cuda"
            return _orig_is_autocast_enabled(device_type)

        torch.is_autocast_enabled = _patched_is_autocast_enabled
