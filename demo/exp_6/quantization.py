import torch
import torch.nn as nn


def enable_4bit(model):
    try:
        from bitsandbytes.nn import Linear4bit
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad = False
    except ImportError:
        pass


def get_compression_ratio(original_size, compressed_size):
    if compressed_size == 0:
        return 0.0
    return original_size / compressed_size
