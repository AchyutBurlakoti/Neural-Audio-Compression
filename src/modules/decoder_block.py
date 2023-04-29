from torch import nn
from .conv_transpose1d_builder import Conv1DT
from .residual import ResidualBlock
from src.config_file import configuration
from .rrdb_residual import RRDB

class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super().__init__()
        
        self.elu = nn.ELU(alpha=1.0)
        self.convtr = Conv1DT(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride)
        
        if configuration["decoder"]["residual_block"] == "simple":
            self.resblock = ResidualBlock(in_channels=out_channels, hidden_channels=out_channels//2, dilation=dilation)
        else :
            self.resblock = RRDB(in_channels=out_channels, hidden_channels=out_channels//2, dilation=dilation)
        
        self.layers = nn.Sequential(
            self.elu,
            self.convtr,
            self.resblock
        )
        
    def forward(self, x):
        return self.layers(x)