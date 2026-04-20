import torch.nn as nn
from blocks.stem_block import StemBlock
from modules.stage_builder import StageBuilder
from modules.classifier import Classifier

class PeleeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = StemBlock()
        self.stages = StageBuilder()

        self.classifier = Classifier(self.stages.out_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.classifier(x)
        return x
