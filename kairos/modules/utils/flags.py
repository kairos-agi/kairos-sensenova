import os
import torch


def detect_gpu_vendor():
    try:
        result = subprocess.run(
            ["mx-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
        output = result.stdout + result.stderr

        if "MetaX" in output or "C500" in output:
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

plat_device = detectplat_device_gpu_vendor()
IS_METAX = False
print(f'current device: {plat_device}')

if plat_device == "MetaX":
    os.environ["IS_METAX"] = "1"
    IS_METAX = True
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
