import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from networks import blocks


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
        

class CoordConv2d(nn.Module):
    def __init__(self, conv, in_channels, out_channels, with_r=True, use_cuda=True):
        super(CoordConv2d, self).__init__()
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = conv(in_channels + self.rank + int(with_r), out_channels, 1)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = F.relu(self.conv(out))
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, num_features):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)       


class Attention(nn.Module):
    def __init__(self, conv, num_features):
        super(Attention, self).__init__()
        self.concat = conv(num_features*2, num_features, 1)
        self.channel = ChannelAttention(num_features, num_features)
        self.spatial = CoordConv2d(conv, num_features, num_features)

    def forward(self, x):
        channel = self.channel(x)
        spatial = self.spatial(x)
        res = self.concat(torch.cat([channel, spatial], dim=1))
        return res


class Block(nn.Module):
    def __init__(self, conv, in_features, num_features):
        super(Block, self).__init__()
        self.conv31 = conv(num_features, num_features, 3)
        self.conv32 = conv(num_features, num_features, 3)
        self.relu = nn.ReLU(inplace=True)
        self.attention = Attention(conv, num_features)

    def forward(self, x):
        out = self.relu(self.conv31(x))
        out = self.conv32(out)
        out = self.attention(out)
        out = out + x
        return out      


class GraphNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_features, upscale_factor,
         res_scale=1, conv=blocks.default_conv):
        super(GraphNet, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

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
        self.concat = conv(num_features*21, num_features, 1)
        self.tail = nn.Sequential(*net_tail)

        self.add_mean = blocks.MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, input):
        b, c, h, w = input.size()
        x = self.sub_mean(input)
        x = self.head(x)
        res = x

        inter_out = []
        for i in range(20):
            x = self.body[i](x)
            inter_out.append(x)
        inter_out.append(res)
        res = torch.cat(inter_out, dim=1)
        res = self.concat(res)
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