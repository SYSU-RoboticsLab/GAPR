import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List

def get_datetime():
    return time.strftime("%Y%m%d_%H%M%S")

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def get_idx_from_string(elem):
    """
    000021.npy -> 21
    """
    return int(elem.split(".")[0])


def tensors2numbers(data):
    """
    ```python
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    ```
    """
    if data is None: return data
    else:
        if torch.is_tensor(data):
            return data.item()
        elif isinstance(data, list) or isinstance(data, tuple):
            for i, _ in enumerate(data):
                data[i] = tensors2numbers(data[i])
            return data
        elif isinstance(data, dict):
            for e in data:
                data[e]  = tensors2numbers(data[e])
            return data
        else:
            return data

def tensors2device(data:Any, device:torch.device):
    """
    # [tensor.to(device)]
    """
    if data is None: return data
    else:
        if torch.is_tensor(data):
            return data.to(device)
        elif isinstance(data, list) or isinstance(data, tuple):
            for i, _ in enumerate(data):
                data[i] = tensors2device(data[i], device)
            return data
        elif isinstance(data, dict):
            for e in data:
                data[e] = tensors2device(data[e], device)
            return data
        else:
            raise NotImplementedError("tensors2device: %s not implemented error"%str(type(data)))


def avg_stats(stats:List):
    avg = stats[0]
    for e in avg:
        if isinstance(avg[e], Dict):
            this_stats = [stats[i][e] for i in range(len(stats))]
            avg[e] = avg_stats(this_stats)
        else:
            avg[e] = np.mean([stats[i][e] for i in range(len(stats))])
    return avg