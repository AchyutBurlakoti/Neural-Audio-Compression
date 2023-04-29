from torch import nn

class LSTM(nn.Module):
    
    def __init__(self, dim, num_layers = 2, skip = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dim, dim, num_layers)
        
    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y