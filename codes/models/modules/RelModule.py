import torch
import torch.nn as nn
import torch.nn.functional as F
from .DenseBlock import DenseBlock

class RelModule(nn.Module):
    def __init__(self, rel_opt):
        super(RelModule, self).__init__()
        self.rel_opt = rel_opt
        if self.rel_opt['use_rel']:
            self.forw_rel = RelBlock(self.rel_opt['num_of_frames'], self.rel_opt['nf'])
            self.back_rel = RelBlock(self.rel_opt['num_of_frames'], self.rel_opt['nf'])

    def forward(self, x, rev=False):
        if not self.rel_opt['use_rel']:
            return x
        if not rev:
            return self.forw_rel(x)
        else:
            return self.back_rel(x)


class RelBlock(nn.Module):
    def __init__(self, num_of_frames, nf):
        super(RelBlock, self).__init__()
        self.num_of_frames = num_of_frames
        self.DB_heads = nn.ModuleList()
        self.conv_1x1_heads = nn.ModuleList()
        self.conv_merges = nn.ModuleList()
        self.conv_1x1_tails = nn.ModuleList()
        self.DB_tails = nn.ModuleList()
        for i in range(self.num_of_frames):
            self.DB_heads.append(DenseBlock(3, nf))
            self.conv_1x1_heads.append(nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True))
            self.conv_merges.append(nn.Conv2d(nf * self.num_of_frames, nf, kernel_size=3, stride=1, padding=1, bias=True))
            self.conv_1x1_tails.append(nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True))
            self.DB_tails.append(DenseBlock(nf, 3))

    def forward(self, x):
        b,c,h,w = x.size()
        x = x.view(b,c//3,3,h,w)
        out_head = []
        for i in range(self.num_of_frames):
            out_i = x[:,i,...]
            out_i = self.DB_heads[i](out_i)
            out_i = self.conv_1x1_heads[i](out_i)
            out_head.append(out_i)
        out_head = torch.cat(out_head, dim=1)

        out_tail = []
        for i in range(self.num_of_frames):
            out_i = self.conv_merges[i](out_head)
            out_i = self.conv_1x1_tails[i](out_i)
            out_i = self.DB_tails[i](out_i)
            out_tail.append(out_i)
        out_tail = torch.cat(out_tail, dim=1)

        x = x.view(b,c,h,w)
        return x + out_tail * 0.2