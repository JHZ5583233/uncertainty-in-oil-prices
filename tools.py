import torch

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    elif torch.xpu.is_available():
        return "xps"

    return "cpu"
