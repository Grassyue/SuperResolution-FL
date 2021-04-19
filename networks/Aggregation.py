import torch
import torch.nn as nn
from torch.nn import functional as F
from networks import blocks
from networks import common


class BasicBlock(nn.Module):
    def __init__(self, conv, in_channels, num_features, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, num_features, 3)]
        if bn:
            m.append(nn.BatchNorm2d(num_features))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, in_channels, num_features, bn=False, act=nn.ReLU(True), res_scale=0.1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(in_channels, num_features, 3, bias=True))
            if bn:
                m.append(nn.BatchNorm2d(num_features))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        # import pdb; pdb.set_trace()
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class GroupNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_features, upscale_factor, conv = blocks.default_conv):
        super(GroupNet, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        n_resblocks = 8
        self.sub_mean = blocks.MeanShift(rgb_mean, rgb_std, sign=-1)
        self.add_mean = blocks.MeanShift(rgb_mean, rgb_std, sign=1)

        #define network head
        self.head = conv(in_channels, num_features, 3)

        #define network body
        self.block1 = ResBlock(conv, num_features, num_features)
        self.block2 = ResBlock(conv, num_features, num_features)
        self.node1 = conv(num_features*2, num_features, 1)
        
        self.block3 = ResBlock(conv, num_features, num_features)
        self.block4 = ResBlock(conv, num_features, num_features)
        self.node2 = conv(num_features*2, num_features, 1)

        self.block5 = ResBlock(conv, num_features, num_features)
        self.block6 = ResBlock(conv, num_features, num_features)      
        self.node3 = conv(num_features*4, num_features*2, 1)

        self.block7 = ResBlock(conv, num_features*2, num_features*2)
        self.block8 = ResBlock(conv, num_features*2, num_features*2)     
        self.node4 = conv(num_features*4, num_features*2, 1)

        self.block9 = ResBlock(conv, num_features*2, num_features*2)
        self.block10 = ResBlock(conv, num_features*2, num_features*2)      
        self.node5 = conv(num_features*2*3, num_features*3, 1)

        self.block11 = ResBlock(conv, num_features*3, num_features*3)
        self.block12 = ResBlock(conv, num_features*3, num_features*3)
        self.node6 = conv(num_features*2*3, num_features*3, 1)

        self.block13 = ResBlock(conv, num_features*3, num_features*3)
        self.block14 = ResBlock(conv, num_features*3, num_features*3)     
        self.node7 = conv(num_features*15, 480, 1)

        #define network tail
        network_tail_1 = [
            conv(480, num_features, 3),
            blocks.Upsampler(conv, upscale_factor, num_features),
            conv(num_features, out_channels, 3)]

        network_tail_2 = [
            blocks.Upsampler(conv, upscale_factor, num_features),
            conv(num_features, out_channels, 3)]

        self.tail_1 = nn.Sequential(*network_tail_1)
        self.tail_2 = nn.Sequential(*network_tail_2)


    def forward(self, x):
        x = self.sub_mean(x)
        first = self.head(x)
        out1 = self.block1(first)
        out2 = self.block2(out1)
        in1 = torch.cat([out1, out2], 1)
        agg1 = self.node1(in1)

        out3 = self.block3(agg1)
        out4 = self.block4(out3)
        in2 = torch.cat([out3, out4], 1)
        agg2 = self.node2(in2)

        out5 = self.block5(agg2)
        out6 = self.block6(out5)
        in3 = torch.cat([out5, out6, agg1, agg2], 1)
        agg3 = self.node3(in3)

        out7 = self.block7(agg3)
        out8 = self.block8(out7)
        in4 = torch.cat([out7, out8], 1)
        agg4 = self.node4(in4)

        out9 = self.block9(agg4)
        out10 = self.block10(out9)
        in5 = torch.cat([agg4, out9, out10], 1)
        agg5 = self.node5(in5)

        out11 = self.block11(agg5)
        out12 = self.block12(out11)
        in6 = torch.cat([out11, out12], 1)
        agg6 = self.node6(in6)

        out13 = self.block13(agg6)
        out14 = self.block14(out13)
        in7 = torch.cat([out13, out14, agg6, agg5, agg3, agg1], 1)
        agg7 = self.node7(in7)      
        
        res = self.tail_1(agg7)
        out = self.tail_2(first)
        HR = torch.add(out,res) 
        HR = self.add_mean(HR) 
    
