from torch import nn
from .conv1d_builder import Conv1D

from src.config_file import configuration

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, dilation=1, norm='weight_norm', causal=False,
                pad_mode='reflect', compress=2, true_skip=True):
        super().__init__()
        
        self.elu1 = nn.ELU(alpha=1.0)
        self.conv1 = Conv1D(in_channels=in_channels, 
                            out_channels=hidden_channels,
                            kernel_size=3,
                            dilation=dilation,
                            stride=1)
        self.elu2 = nn.ELU(alpha=1.0)
        self.conv2 = Conv1D(in_channels=hidden_channels, 
                            out_channels=in_channels,
                            kernel_size=1,
                            dilation=1,
                            stride=1)
        if true_skip:
            self.shortcut = nn.Identity()
        else :
            self.shortcut = Conv1D(in_channels, in_channels, kernel_size=1)

        self.layers = None

        if configuration["dropout"] :
            self.dropout1 = nn.Dropout1d(0.2)
            self.dropout2 = nn.Dropout1d(0.2)

            self.layers = nn.Sequential(
                self.conv1,
                self.elu1,
                self.dropout1,
                self.conv2,
                self.elu2,
                self.dropout2
            )

        else :
            
            self.layers = nn.Sequential(
                self.elu1,
                self.conv1,
                self.elu2,
                self.conv2
            )
    
    def forward(self, x):
        return self.shortcut(x) + self.layers(x)