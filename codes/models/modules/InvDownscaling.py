import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InvDownscaling(nn.Module):
    def __init__(self, down_opt):
        super(InvDownscaling, self).__init__()
        self.down_opt = down_opt
        if self.down_opt['use_down']:
            self.in_nc = self.down_opt['in_nc']
            self.order = self.down_opt['order']
            self.operations = nn.ModuleList()
            if self.down_opt['scale'] > 1:
                if self.down_opt['type'] == 'haar':
                    b = HaarDownsampling(self.in_nc, self.order)
                    self.operations.append(b)
                elif self.down_opt['type'] == 'squeeze':
                    b = SqueezeLayer(2, self.order)
                    self.operations.append(b)
                self.in_nc = self.in_nc * self.down_opt['scale'] * self.down_opt['scale']
                if self.down_opt['use_conv1x1']:
                    b = InvertibleConv1x1(self.in_nc)
                    self.operations.append(b)

    def forward(self, x, rev=False):
        if not self.down_opt['use_down']:
            return x
        if not rev:
            for op in self.operations:
                x = op.forward(x, rev)
        else:
            for op in reversed(self.operations):
                x = op.forward(x, rev)
        return x


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in, order):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

        self.order = order
        if self.order == 'ref':
            self.forw_order = [i * self.channel_in // 3 + j for j in range(self.channel_in // 3) for i in range(4)]
            self.back_order = [i + j * 4 for i in range(4) for j in range(self.channel_in // 3)]
        elif self.order == 'hl':
            pass

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            
            if self.order == 'ref':
                out = out.reshape([x.shape[0], self.channel_in * 4 // 3, 3, x.shape[2] // 2, x.shape[3] // 2])
                out = out[:, self.forw_order, :, :, :]
                out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            elif self.order == 'hl':
                pass
            return out
        else:
            if self.order == 'ref':
                out = x.reshape([x.shape[0], self.channel_in * 4 // 3, 3, x.shape[2], x.shape[3]])
                out = out[:, self.back_order, :, :, :]
                out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            elif self.order == 'hl':
                out = x

            out = out.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)


class SqueezeLayer(nn.Module):
    def __init__(self, factor, order):
        super().__init__()
        self.factor = factor
        self.order = order

    def forward(self, input, rev=False):
        if not rev:
            output = self.squeeze2d(input, self.factor, self.order)  # Squeeze in forward
            return output
        else:
            output = self.unsqueeze2d(input, self.factor, self.order)
            return output
        
    @staticmethod
    def squeeze2d(input, factor, order):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, factor * factor * C, H // factor, W // factor)
        
        if order == 'ref':
            forw_order = [i * C // 3 + j for j in range(C // 3) for i in range(4)]
            x = x.reshape([B, C * 4 // 3, 3, H // 2, W // 2])
            x = x[:, forw_order, :, :, :]
            x = x.reshape([B, C * 4, H // 2, W // 2])
        elif order == 'hl':
            pass
        return x

    @staticmethod
    def unsqueeze2d(input, factor, order):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor2) == 0, "{}".format(C)

        if order == 'ref':
            back_order = [i + j * 4 for i in range(4) for j in range(C // factor2 // 3)]
            x = input.reshape([B, C // 3, 3, H, W])
            x = x[:, back_order, :, :, :]
            x = x.reshape([B, C, H, W])
        elif order == 'hl':
            x = input
        x = x.view(B, factor, factor, C // factor2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, input, rev):
        w_shape = self.w_shape
        if not rev:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, input, rev=False):
        weight = self.get_weight(input, rev)
        if not rev:
            z = F.conv2d(input, weight)
            return z
        else:
            z = F.conv2d(input, weight)
            return z