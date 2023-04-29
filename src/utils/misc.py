from torch import nn
import torch

from torch.nn.utils import spectral_norm, weight_norm
from torch.nn import functional as F
import math
import typing as tp

def get_norm_module(module, norm='norm'):
      return nn.Identity()

def apply_parametrization_norm(module: nn.Module):
    return weight_norm(module)

def get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total=0):

    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length

def unpad1d(x, paddings):

    padding_left, padding_right = paddings
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]

def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)