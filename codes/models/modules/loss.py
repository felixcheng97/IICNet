import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0


class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, x, target):
        b, c, h, w = x.size()
        x = x.contiguous().view(-1,h,w)
        target = target.contiguous().view(-1,h,w)
        x_fft = torch.rfft(x, signal_ndim=2, normalized=False, onesided=True)
        target_fft = torch.rfft(target, signal_ndim=2, normalized=False, onesided=True)
        
        _, h, w, f = x_fft.size()
        x_fft = x_fft.view(b,c,h,w,f)
        target_fft = target_fft.view(b,c,h,w,f)
        diff = x_fft - target_fft
        return torch.mean(torch.sum(diff**2, (1, 2, 3, 4)))
