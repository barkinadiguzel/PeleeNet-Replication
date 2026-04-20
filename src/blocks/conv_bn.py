import torch
import torch.nn as nn

class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))
