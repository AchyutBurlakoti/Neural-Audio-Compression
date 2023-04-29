from torch import nn
import torch
from .conv1d_builder import Conv1D
from src.config_file import configuration

class Block(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, dilation=1, norm='weight_norm', causal=False,
                pad_mode='reflect', compress=2, true_skip=True):
        super().__init__()
        
        self.conv1 = Conv1D(in_channels=in_channels, 
                            out_channels=hidden_channels,
                            kernel_size=3,
                            dilation=dilation,
                            stride=1)
        self.elu1 = nn.ELU(alpha=1.0)

        self.layers = None

        if configuration["dropout"] :
          self.dropout = nn.Dropout1d(0.2)
            
          self.layers = nn.Sequential(
              self.conv1,
              self.elu1,
              self.dropout
          )

        else :
          self.layers = nn.Sequential(
             self.conv1,
             self.elu1
          )
    
    def forward(self, x):
        return self.layers(x)

class DenseBlock(nn.Module):

  def __init__(self, in_channels, hidden_channels, dilation=1, norm='weight_norm', causal=False):
    super().__init__()

    self.res1 = Block(in_channels=in_channels, hidden_channels=in_channels, dilation=dilation)
    self.res2 = Block(in_channels=2*in_channels, hidden_channels=in_channels, dilation=dilation)
    self.res3 = Block(in_channels=3*in_channels, hidden_channels=in_channels, dilation=dilation)
    self.res4 = Block(in_channels=4*in_channels, hidden_channels=in_channels, dilation=dilation)

    self.blocks = [self.res1, self.res2]

    self.conv1 = Conv1D(in_channels=in_channels, out_channels=in_channels, kernel_size=1, dilation=1, stride=1)

    self.shortcut = nn.Identity()

    self.res_scale = 0.2

  def forward(self, x):

    inputs = x
    for block in self.blocks:
      out = block(inputs)
      inputs = torch.cat([inputs, out], 1)

    return out.mul(self.res_scale) + x

class RRDB(nn.Module):

  def __init__(self, in_channels, hidden_channels, dilation=1, norm='weight_norm', causal=False):
    super().__init__()

    self.dense1 = DenseBlock(in_channels=in_channels, hidden_channels=hidden_channels, dilation=dilation)
    self.dense2 = DenseBlock(in_channels=in_channels, hidden_channels=hidden_channels, dilation=dilation)
    self.dense3 = DenseBlock(in_channels=in_channels, hidden_channels=hidden_channels, dilation=dilation)

    self.beta = 0.2

    self.dense_blocks = nn.Sequential(self.dense1, self.dense2, self.dense3)

  def forward(self, x):
    return self.dense_blocks(x).mul(self.beta) + x