import torch
import torch.nn as nn

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 255.)
        output = input.round() * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Round(nn.Module):
    def __init__(self):
        super(Round, self).__init__()

    def forward(self, input, **kwargs):
        return Quant.apply(input * 255.0) / 255.0
