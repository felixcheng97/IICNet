import torch
import torch.nn as nn

class CSLayer(nn.Module):
    def __init__(self, cs_opt):
        super(CSLayer, self).__init__()
        self.cs_opt = cs_opt

    def forward(self, x, rev=False):
        if not rev:
            output_frames = x
            b,c,h,w = output_frames.size()
            out_nc = self.cs_opt['out_nc']
            embedding_frame = torch.mean(output_frames.view(b,c//out_nc,out_nc,h,w), dim=1)
            return embedding_frame
        else:
            embedding_frame_prime = x
            times = self.cs_opt['in_nc'] // self.cs_opt['out_nc']
            output_frames_prime = embedding_frame_prime.repeat(1,times,1,1)
            return output_frames_prime

            