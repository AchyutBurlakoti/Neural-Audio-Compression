from torch import nn
from src.modules.decoder_block import DecoderBlock
from src.modules.conv1d_builder import Conv1D
from src.modules.lstm import LSTM

class Decoder(nn.Module):
    
    def __init__(self, C=128, D=32):
        super().__init__()
        
        self.conv1 = Conv1D(C, 16 * D, kernel_size=7, stride=1, norm="weight", causal=False, pad_mode='reflect')
        self.lstm = LSTM(4*C)
        
        self.dec1 = DecoderBlock(in_channels=16*D, out_channels=8*D, kernel_size=16, stride=8, dilation=1)
        self.dec2 = DecoderBlock(in_channels=8*D, out_channels=4*D, kernel_size=10, stride=5, dilation=2)        
        self.dec3 = DecoderBlock(in_channels=4*D, out_channels=2*D, kernel_size=8, stride=4, dilation=4)        
        self.dec4 = DecoderBlock(in_channels=2*D, out_channels=D, kernel_size=4, stride=2, dilation=8) 
        
        self.elu = nn.ELU(alpha=1.0)
        self.conv2 = Conv1D(in_channels=D, out_channels=1, kernel_size=7)
        
        self.layers = nn.Sequential(
            self.conv1,
            self.lstm,
            self.dec1,
            self.dec2,
            self.dec3,
            self.dec4,
            self.elu,
            self.conv2
        )
        
    def forward(self, x):
        return self.layers(x)