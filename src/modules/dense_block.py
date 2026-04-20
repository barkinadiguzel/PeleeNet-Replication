import torch.nn as nn
from blocks.two_way_dense import TwoWayDense

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_ch, growth_rate):
        super().__init__()

        layers = []
        ch = in_ch

        for _ in range(num_layers):
            layers.append(TwoWayDense(ch, growth_rate))
            ch += growth_rate * 2

        self.block = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x):
        return self.block(x)
