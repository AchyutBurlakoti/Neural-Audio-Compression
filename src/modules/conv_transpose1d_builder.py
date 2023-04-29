from torch import nn
from torch.nn.utils import spectral_norm, weight_norm
import math

from src.utils.misc import *

class Conv1DT(nn.Module):
    
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True, causal=False,
                             norm='none', pad_mode='reflect'):
        super().__init__()
        
        self.convtr = weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, 
                                  padding_mode='zeros'))
        self.pad_mode = pad_mode
        self.causal = causal
        
        self.norm = nn.Identity()
        
    def forward(self, x):
        
        kernel_size = self.convtr.kernel_size[0]
        stride = self.convtr.stride[0]
        
        padding_total = kernel_size - stride
        y = self.norm(self.convtr(x))
                
        if self.causal:
            padding_right = math.ceil(padding_total * 1.0)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else :
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y