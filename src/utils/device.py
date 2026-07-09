import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
