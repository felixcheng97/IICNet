import torch
import torch.nn as nn
from .InvArch import InvArch

class InvBlock(nn.Module):
    def __init__(self, block_opt):
        super(InvBlock, self).__init__()
        self.block_opt = block_opt

        self.operations = nn.ModuleList()
        for _ in range(self.block_opt['block_num']):
            b = InvArch(self.block_opt['type'], self.block_opt['split_len1'], self.block_opt['split_len2'])
            self.operations.append(b)

    def forward(self, x, rev=False):
        if not rev:
            for op in self.operations:
                x = op.forward(x, rev)
        else:
            for op in reversed(self.operations):
                x = op.forward(x, rev)
        return x


    