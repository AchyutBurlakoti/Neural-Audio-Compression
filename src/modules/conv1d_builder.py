from torch import nn
from torch.nn.utils import spectral_norm, weight_norm
from src.utils.misc import *

class Conv1D(nn.Module):
    
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, 
                              dilation=1, groups=1, bias=True, causal=False,
                             norm='none', pad_mode='reflect'):
        super().__init__()
        
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                                dilation=dilation, groups=1, bias=True, padding_mode=pad_mode))
        self.pad_mode = pad_mode
        self.causal = causal
        
        self.norm = nn.Identity()
        
    def forward(self, x):
        
        B, C, T = x.shape
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        
        padding_total = (kernel_size - 1) * dilation - (stride - 1)
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else :
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        x = self.conv(x)
        return self.norm(x)