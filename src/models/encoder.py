from torch import nn
from src.modules.encoder_block import EncoderBlock
from src.modules.conv1d_builder import Conv1D
from src.modules.lstm import LSTM

class Encoder(nn.Module):
    
    def __init__(self, C=32, D=128):
        super().__init__()
        
        self.conv1 = Conv1D(1, C, kernel_size=7, stride=1, norm="weight", causal=False, pad_mode='reflect')
        
        self.enc1 = EncoderBlock(in_channels=C, out_channels=2*C, kernel_size=4, stride=2, dilation=1)
        self.enc2 = EncoderBlock(in_channels=2*C, out_channels=4*C, kernel_size=8, stride=4, dilation=2)        
        self.enc3 = EncoderBlock(in_channels=4*C, out_channels=8*C, kernel_size=10, stride=5, dilation=4)        
        self.enc4 = EncoderBlock(in_channels=8*C, out_channels=16*C, kernel_size=16, stride=8, dilation=8) 
        
        if True:
            self.lstm = LSTM(16*C)
        
        self.elu = nn.ELU(alpha=1.0)
        self.conv2 = Conv1D(16*C, D, kernel_size=7, stride=1)
        
        self.layers = nn.Sequential(
            self.conv1,
            self.enc1,
            self.enc2,
            self.enc3,
            self.enc4,
            self.lstm,
            self.elu,
            self.conv2
        )
        
    def forward(self, x):
        return self.layers(x)