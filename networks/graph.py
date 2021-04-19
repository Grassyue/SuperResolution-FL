import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from networks import blocks

import numpy as np

# @torch.no_grad()
def compute_adj(x):
    b, c, h, w = x.size()
    features = x.view(b, c, h*w)
    sim = torch.bmm(features.permute(0,2,1), features)
    sim = sim.div(b*c*h*w)
    sim = torch.relu(sim)
    d = torch.rsqrt(torch.sum(sim, dim=-1))
    d2 = torch.diag_embed(d)
    adj = torch.matmul(d2, sim)
    adj = torch.matmul(adj, d2)
    return adj


class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError
        return out


# class CoordConv2d(nn.Module):
#     def __init__(self, conv, in_channels, out_channels, with_r=True, use_cuda=True):
#         super(CoordConv2d, self).__init__()
#         self.rank = 2
#         self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
#         self.conv = conv(in_channels + self.rank + int(with_r), out_channels, 1)

#     def forward(self, input_tensor):
#         out = self.addcoords(input_tensor)
#         out = self.conv(out)
#         return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_features//reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(num_features//reduction, num_features, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x*y


# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, num_features):0
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3-1)//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)     


class GraphConv_Layer(nn.Module):
    def __init__(self, conv, in_features, num_features, bias=True):
        super(GraphConv_Layer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, num_features))
        nn.init.normal_(self.weight.data)
        self.conv = conv(num_features, num_features, 3)
        self.concat = conv(num_features*2, num_features, 1)
        
    def forward(self, x, adj):
        b, c, h, w = x.size()
        local = self.conv(x)
        x_re = x.view(b, c, h*w)
        support = torch.matmul(x_re.permute(0,2,1), self.weight)
        non_local = torch.matmul(adj, support)
        non_local = non_local.view(b, c, h, w)
        res = self.concat(torch.cat([local, non_local], dim=1))
        return res


class SpatialAttention(nn.Module):
    def __init__(self, conv, in_features, num_features):
        super(SpatialAttention, self).__init__()
        self.graph1 = GraphConv_Layer(conv, in_features, num_features)
        self.bn = nn.BatchNorm2d(num_features)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, adj):
        res = self.graph1(x, adj)
        res = self.act(self.bn(res))
        res = x * res
        return res


class Attention(nn.Module):
    def __init__(self, conv, num_features):
        super(Attention, self).__init__()
        self.concat = conv(num_features*2, num_features, 1)
        self.channel = ChannelAttention(num_features, num_features)
        self.spatial = SpatialAttention(conv, num_features, num_features)

    def forward(self, x, adj):
        channel = self.channel(x)
        spatial = self.spatial(x, adj)
        res = self.concat(torch.cat([channel, spatial], dim=1))
        return res


class Block(nn.Module):
    def __init__(self, conv, in_features, num_features):
        super(Block, self).__init__()
        self.conv31 = conv(num_features, num_features, 3)
        self.conv32 = conv(num_features*2, num_features*2, 3)
        self.conv51 = conv(num_features, num_features, 5)
        self.conv52 = conv(num_features*2, num_features*2, 5)
        self.concat = conv(num_features*4, num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        self.attention = Attention(conv, num_features)

    def forward(self, x, adj):
        out = x
        out31 = self.relu(self.conv31(out))
        out51 = self.relu(self.conv51(out))
        out = torch.cat([out31, out51], dim=1)
        out32 = self.relu(self.conv32(out))
        out52 = self.relu(self.conv52(out))
        out = self.concat(torch.cat([out32, out52], dim=1))
        out = self.attention(out, adj)
        out = out + x
        return out      


class Graph(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_features, upscale_factor,
         res_scale=1, conv=blocks.default_conv):
        super(Graph, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        ##define adjency matrix
        self.pos = AddCoords(rank=2, with_r=True)
        self.concat1 = conv(num_features+3, num_features, 1)

        ##define network head
        net_head = [conv(in_channels, num_features, 3)]

        ##define network body
        net_body = nn.ModuleList()
        for i in range(20):
            net_body.append(Block(conv, num_features, num_features))

        ##deine network tail
        net_tail = [
            conv(num_features, num_features, 3),
            blocks.Upsampler(conv, upscale_factor, num_features),
            conv(num_features, out_channels=3, kernel_size=3)
        ]  

        self.sub_mean = blocks.MeanShift(rgb_mean, rgb_std, sign=-1)
        self.head = nn.Sequential(*net_head)
        self.body = nn.Sequential(*net_body)
        self.concat2 = conv(num_features*21, num_features, 1)
        self.tail = nn.Sequential(*net_tail)
        self.add_mean = blocks.MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, input):
        b, c, h, w = input.size()
        x = self.sub_mean(input)
        x = self.head(x)
        res = x
        pos = self.pos(res)
        x_pos = self.concat1(pos)
        # x_norm = nn.functional.normalize(x, p=1/2, dim=-1)
        adj = compute_adj(x_pos)

        inter_out = []
        for i in range(20):
            x = self.body[i](x, adj)
            inter_out.append(x)
        inter_out.append(res)
        res = torch.cat(inter_out, dim=1)
        res = self.concat2(res)
        x = self.tail(res)
        x = self.add_mean(x)
        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

