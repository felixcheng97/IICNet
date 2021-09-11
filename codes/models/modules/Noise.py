import torch
import torch.nn as nn

class Noise(nn.Module):
    def __init__(self):
        super(Noise, self).__init__()

    def forward(self, input, train=False):
        input = input * 255.0
        if train:
            noise = torch.nn.init.uniform_(torch.zeros_like(input), -0.5, 0.5).cuda()
            output = input + noise
            output = torch.clamp(output, 0, 255.)
        else:
            output = input.round() * 1.0
            output = torch.clamp(output, 0, 255.)
        return output / 255.0
