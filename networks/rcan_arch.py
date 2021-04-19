import torch
import torch.nn as nn
from networks import blocks
import math 
import torch.nn.functional as F

# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y

def resolution(x):
    return x.size()[-2:]

class MHSA(nn.Module):
    def __init__(self, conv, num_features):
        super(MHSA, self).__init__()
        self.query = conv(num_features, num_features, 1)
        self.key = conv(num_features, num_features, 1)
        self.value = conv(num_features, num_features, 1)
        self.softmax = nn.Softmax(dim=-1)

    def resize(self, x, num_features):
        height, weight = x.size()[-2:]
        self.rel_h = nn.Parameter(torch.randn([1, num_features, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, num_features, weight, 1]), requires_grad=True)
        return self.rel_h, 



# class RCAB(nn.Module):
#     def __init__(
#         self, conv, n_feat, kernel_size, reduction,
#         bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

#         super(RCAB, self).__init__()
#         modules_body = []
#         for i in range(2):
#             modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
#             if i == 0: modules_body.append(act)
#         modules_body.append(CALayer(n_feat, reduction))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale

#     def forward(self, x):
#         res = self.body(x)
#         #res = self.body(x).mul(self.res_scale)
#         res += x
#         return res

# class ResidualGroup(nn.Module):
#     def __init__(self, conv, num_features, kernel_size, reduction, res_scale, n_resblocks ,act_type='relu') :
#         super(ResidualGroup, self).__init__()

#         modules_body = []
#         modules_body = [
#             RCAB(conv, num_features, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)            
#             for i in range(n_resblocks)
#         ]
#         modules_body.append(conv(num_features, num_features, kernel_size))
#         self.body = nn.Sequential(*modules_body)

#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res

class RCAN(nn.Module):
    def __init__(
        self, num_features, n_resgroups, n_resblocks, kernel_size, reduction, upscale_factor,
        norm_type=False, act_type='relu', conv=blocks.default_conv
    ):
        super(RCAN, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = blocks.MeanShift(rgb_mean, rgb_std)

        modules_head = [conv(in_channels=3, out_channels=num_features, kernel_size=kernel_size)]

        modules_body = [
            ResidualGroup(
                conv, num_features, kernel_size, reduction, res_scale=1, n_resblocks=20, act_type='relu'
            )
            for i in range(n_resgroups)
        ]

        modules_body.append(conv(num_features, num_features, kernel_size))
        modules_tail = [
            blocks.Upsampler(conv, upscale_factor, num_features, act_type=False),
            conv(num_features, out_channels=3, kernel_size=kernel_size)
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.add_mean = blocks.MeanShift(rgb_mean, rgb_std, sign=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        hr = self.add_mean(x)
        return hr

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


