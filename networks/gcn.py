### pyramid pooling in GCN
import math
import torch
import torch.nn as nn
from torch import einsum
from torch.nn import functional as F
from networks import blocks

# def cal_adj_matrix(x):
#     b, c, h, w = x.size()
#     feature = x.view(b, c, h*w)
#     sim = torch.bmm(feature.permute(0, 2, 1), feature)
#     sim = sim.div(b*c*h*w)
#     sim = torch.relu(sim)
#     d1 = torch.rsqrt(torch.sum(sim, dim=-1))
#     d2 = torch.diag_embed(d1)
#     adj = torch.matmul(d2, sim)
#     adj = torch.matmul(adj, d2)
#     return adj

##############################################################################################
### pyramid pooling on feature

class Pyramid_pool_feature(nn.Module):
    def __init__(self):
        super(Pyramid_pool_feature, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

    def forward(self, x):
        b, c, h, w = x.size()
        x_pool_1 = self.pool1(x).view(b,c,-1)
        x_pool_2 = self.pool2(x).view(b,c,-1)
        x_pool_3 = self.pool3(x).view(b,c,-1)
        x_pool_4 = self.pool4(x).view(b,c,-1)
        x_pool = torch.cat([x_pool_1, x_pool_2, x_pool_3, x_pool_4], -1)
        return x_pool
##############################################################################################

##############################################################################################
### pyramid pooling in adj_matrix

class Cal_adj_matrix(nn.Module):
    def __init__(self):
        super(Cal_adj_matrix, self).__init__()
        self.pool = Pyramid_pool_feature()

    def forward(self, x):
        b, c, h, w = x.size()
        feature = x.view(b, c, -1)
        x_pool = self.pool(x)
        sim = torch.bmm(feature.permute(0,2,1), x_pool)
        sim = sim.div(b*c*h*w)
        sim = torch.relu(sim)
        total = torch.sum(sim, dim=-1)
        d1 = torch.rsqrt(total*total+0.000001)
        d2 = torch.diag_embed(d1)
        adj = torch.einsum('bmm,bmn,bmm->bmn', d2, sim, d2)
        return adj
##############################################################################################


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
        out = self.avg_pool(x)
        out = self.conv(out)
        return x*out


# class GraConvLayer(nn.Module):
#     def __init__(self, conv, in_channels, num_features, bias=True):
#         super(GraConvLayer, self).__init__()
#         self.weight = nn.Parameter(torch.FloatTensor(in_channels, num_features))
#         nn.init.kaiming_normal_(self.weight.data)
#         self.conv = conv(num_features, num_features, 3)
#         self.fusion = conv(2*num_features, num_features, 1)

#     def forward(self, x, adj):
#         b, c, h, w = x.size()
#         local = self.conv(x)
#         x_reshape = x.view(b, c, h*w)
#         support = torch.matmul(x_reshape.permute(0, 2, 1), self.weight)
#         non_local = torch.matmul(adj, support)
#         non_local = non_local.view(b, c, h, w)
#         out = self.fusion(torch.cat([local, non_local], dim=1))
#         return out+x

##############################################################################################
### pyramid pooling in GCN

class GraConvLayer(nn.Module):
    def __init__(self, conv, in_channels, num_features, bias=True):
        super(GraConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, num_features))
        nn.init.kaiming_normal_(self.weight.data)
        self.conv = conv(num_features, num_features, 3)
        self.fusion = conv(2*num_features, num_features, 1)
        self.pool = Pyramid_pool_feature()

    def forward(self, x, adj):
        b, c, h, w = x.size()
        local = self.conv(x)
        x_pool = self.pool(x)
        support = torch.einsum('bcs,cc->bcs', x_pool, self.weight)
        non_local = torch.einsum('bis,bcs->bci', adj, support)
        non_local = non_local.view(b, c, h, w)
        out = self.fusion(torch.cat([local, non_local], dim=1))
        return out+x
##############################################################################################

class GraphAttention(nn.Module):
    def __init__(self, conv, num_features):
        super(GraphAttention, self).__init__()
        self.graph = GraConvLayer(conv, num_features, num_features)
        self.norm = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj):
        # import pdb; pdb.set_trace()
        out = self.graph(x, adj)
        # out1 = self.norm(out)
        out = self.relu(out) + x
        # out = self.relu(self.norm(out))
        return out


class ChannelGraph(nn.Module):
    def __init__(self, conv, num_features):
        super(ChannelGraph, self).__init__()
        self.channel = ChannelAttention(num_features, num_features)
        self.graph = GraphAttention(conv, num_features)
        self.fusion = conv(2*num_features, num_features, 1)

    def forward(self, x, adj):
        channel = self.channel(x)
        graph = self.graph(x, adj)
        out = self.fusion(torch.cat([channel, graph], dim=1))
        return out


class Backbone(nn.Module):
    def __init__(self, conv, num_features):
        super(Backbone, self).__init__()
        self.conv31 = conv(num_features, num_features, 3)
        self.conv32 = conv(num_features*2, num_features*2, 3)
        self.conv51 = conv(num_features, num_features, 5)
        self.conv52 = conv(num_features*2, num_features*2, 5)
        self.fusion = conv(num_features*4, num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        self.attention = ChannelGraph(conv, num_features)        

    def forward(self, x, adj):
        out31 = self.relu(self.conv31(x))
        out51 = self.relu(self.conv51(x))
        out = torch.cat([out31, out51], dim=1)
        out32 = self.relu(self.conv32(out))
        out52 = self.relu(self.conv52(out))
        out = self.fusion(torch.cat([out32, out52], dim=1))
        out = self.attention(out, adj)
        return x+out    


class GCNSR(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, upscale_factor, conv=blocks.default_conv):
        super(GCNSR, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        ## define network head
        self.head = conv(in_channels, num_features, 3)

        ## define network body
        net_body = nn.ModuleList()
        for _ in range(20):
            net_body.append(Backbone(conv, num_features))
        self.body = nn.Sequential(*net_body)

        ## define network tail
        net_tail = [
            conv(num_features, num_features, 3),
            blocks.Upsampler(conv, upscale_factor, num_features),
            conv(num_features, out_channels=3, kernel_size=3)
        ] 
        self.tail = nn.Sequential(*net_tail)

        self.sub_mean = blocks.MeanShift(rgb_mean, rgb_std, sign=-1)
        self.fusion = conv(num_features*21, num_features, 1)
        self.add_mean = blocks.MeanShift(rgb_mean, rgb_std, 1)
        self.adj = Cal_adj_matrix()


    def forward(self, input):
        b, c, h, w = input.size()
        x = self.sub_mean(input)
        x = self.head(x)
        res = x
##############################################################################################
### pyramin pooling in adj_matrix
        adj_first = self.adj(x)
##############################################################################################
        inter_out = []
        for i in range(20):
            x = self.body[i](x, adj_first)
            inter_out.append(x)
        inter_out.append(res)
        out = self.fusion(torch.cat(inter_out, dim=1))
        out = out + res
##############################################################################################
        ### add new adj loss
        adj_last = self.adj(out)
##############################################################################################

        out = self.add_mean(self.tail(out))
        return out, adj_first, adj_last