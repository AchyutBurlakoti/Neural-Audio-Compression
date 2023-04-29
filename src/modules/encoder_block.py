from torch import nn
from .residual import ResidualBlock
from .conv1d_builder import Conv1D
from .rrdb_residual import RRDB

from src.config_file import configuration

class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super().__init__()
        
        if configuration["encoder"]["residual_block"] == "simple":
            self.resblock = ResidualBlock(in_channels=in_channels, hidden_channels=in_channels//2, dilation=dilation)
        else :
            self.resblock = RRDB(in_channels=in_channels, hidden_channels=in_channels//2, dilation=dilation)
        self.elu = nn.ELU(alpha=1.0)
        self.conv = Conv1D(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        
        self.layers = nn.Sequential(
            self.resblock,
            self.elu,
            self.conv
        )
        
    def forward(self, x):
        return self.layers(x)