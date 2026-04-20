import torch.nn as nn
from modules.dense_block import DenseBlock
from blocks.transition_layer import TransitionLayer

class StageBuilder(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage_cfg = [3, 4, 8, 6]
        self.growth = 32

        self.layers = nn.ModuleList()

        in_ch = 32  

        # Stage 1
        self.s1 = DenseBlock(3, in_ch, self.growth)
        in_ch = self.s1.out_ch
        self.t1 = TransitionLayer(in_ch)

        # Stage 2
        self.s2 = DenseBlock(4, in_ch, self.growth)
        in_ch = self.s2.out_ch
        self.t2 = TransitionLayer(in_ch)

        # Stage 3
        self.s3 = DenseBlock(8, in_ch, self.growth)
        in_ch = self.s3.out_ch
        self.t3 = TransitionLayer(in_ch)

        # Stage 4 (no downsample)
        self.s4 = DenseBlock(6, in_ch, self.growth)
        self.out_ch = self.s4.out_ch

    def forward(self, x):
        x = self.s1(x); x = self.t1(x)
        x = self.s2(x); x = self.t2(x)
        x = self.s3(x); x = self.t3(x)
        x = self.s4(x)
        return x
