import torch
import torch.nn as nn
from torch.nn import functional as F
from networks import blocks
from networks import common

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, inChannels,outChannels,reduction=16):
        super(CALayer, self).__init__()
        self.conv1 =nn.Conv2d(inChannels,outChannels,kernel_size=1,padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(outChannels, outChannels//reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(outChannels//reduction, outChannels, 1, padding=0, bias=True),
                nn.Sigmoid()
                )
    def forward(self,x):
        out = self.conv1(x)
        y = self.avg_pool(out)
        y = self.conv_du(y)        
        return y*out


# class BasicBlock(nn.Module):
#     def __init__(self, conv, num_features, bn=True, act=nn.ReLU(True)):
#         m = [conv(num_features, num_features, 3)]
#         if bn:
#             m.append(nn.BatchNorm2d(num_features))
#         if act is not None:
#             m.append(act)
#         super(BasicBlock, self).__init__(*m)


########################################################
# residual block in EDSR
class ResUnit(nn.Module):
    def __init__(self, conv, num_features, bn=False, act=nn.ReLU(True), res_scale=0.1):
        super(ResUnit, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(num_features, num_features, 3, bias=True))
            if bn:
                m.append(nn.BatchNorm2d(num_features))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class BasicUnit(nn.Module):
    def __init__(self, conv, num_features, bn=False, act=nn.ReLU(True), res_scale=0.1):
        super(BasicUnit, self).__init__()
        body = []
        for i in range(2):
            body.append(ResUnit(conv, num_features))
        self.body = nn.Sequential(*body)
    
    def forward(self, x):
        res = self.body(x)
        out = torch.add(res, x)
        return out


########################################################
# dense block in RDN
# class one_conv(nn.Module):
#     def __init__(self,conv,inchanels,growth_rate):
#         super(one_conv,self).__init__()
#         self.conv = conv(inchanels,growth_rate,3)
#         self.relu = nn.ReLU(True)

#     def forward(self,x):
#         output = self.relu(self.conv(x))
#         return torch.cat((x,output),1)

# class BasicUnit(nn.Module):
#     def __init__(self, conv, num_features, c=6, g=32):
#         super(BasicUnit, self).__init__()
#         convs = []
#         for i in range(c):
#             convs.append(one_conv(conv, num_features+i*g, g))
#         self.conv = nn.Sequential(*convs)
#         self.llf = conv(num_features+c*g, num_features, 1)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.llf(out)
#         return out


########################################################
## multi-scale block in MSRN
# class BasicUnit(nn.Module):
#     def __init__(self, conv, n_feats):
#         super(BasicUnit, self).__init__()
#         kernel_size_1 = 3
#         kernel_size_2 = 5
#         self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
#         # self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
#         self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
#         # self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
#         self.confusion = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         input_1 = x
#         output_3_1 = self.relu(self.conv_3_1(input_1))
#         output_5_1 = self.relu(self.conv_5_1(input_1))
#         input_2 = torch.cat([output_3_1, output_5_1], 1)
#         # output_3_2 = self.relu(self.conv_3_2(input_2))
#         # output_5_2 = self.relu(self.conv_5_2(input_2))
#         # input_3 = torch.cat([output_3_2, output_5_2], 1)
#         output = self.confusion(input_2)
#         output += x
#         return output


class GroupNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, n_feats, upscale_factor, conv=blocks.default_conv):
        super(GroupNet, self).__init__()
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = blocks.MeanShift(rgb_mean, rgb_std, sign=-1)
        self.add_mean = blocks.MeanShift(rgb_mean, rgb_std, sign=1)
        
        # define head module
        self.head = conv(in_channels, n_feats, kernel_size=3)

        # define body module                 

        self.unit1 = BasicUnit(conv,n_feats)        
        self.unit2 = BasicUnit(conv,n_feats)        
        self.node1 = conv(n_feats*2,n_feats,1)
        
        self.unit3 = BasicUnit(conv,n_feats)        
        self.unit4 = BasicUnit(conv,n_feats)        
        self.node2 = conv(n_feats*2,n_feats, 1)
        
        self.unit5 = BasicUnit(conv,n_feats)
        self.unit6 = BasicUnit(conv,n_feats)        
        self.node3 = conv(n_feats*4,n_feats*2, 1)
       
        self.unit7 = BasicUnit(conv,n_feats*2)        
        self.unit8 = BasicUnit(conv,n_feats*2)        
        self.node4 = conv(n_feats*4,n_feats*2, 1)
        
        self.unit9 = BasicUnit(conv,n_feats*2)        
        self.unit10 = BasicUnit(conv,n_feats*2)        
        self.node5 = conv(n_feats*2*3,n_feats*3,1)
        
        self.unit11 = BasicUnit(conv,n_feats*3)        
        self.unit12 = BasicUnit(conv,n_feats*3)        
        self.node6 = conv(n_feats*6,n_feats*3,1)
        
        self.unit13 = BasicUnit(conv,n_feats*3)        
        self.unit14 = BasicUnit(conv,n_feats*3)        
        self.node7 = conv(n_feats*15,480, 1)
        
        # define tail module
        modules_tail_1 = [
            conv(480, n_feats, kernel_size=3),
            common.Upsampler(conv, upscale_factor, n_feats, act=False),
            conv(n_feats, out_channels, kernel_size=3)]
            
        modules_tail_2 = [
            common.Upsampler(conv, upscale_factor, n_feats, act=False),
            conv(n_feats, out_channels, kernel_size=3)] 
            
        self.tail_1 =nn.Sequential(*modules_tail_1) 
        self.tail_2 =nn.Sequential(*modules_tail_2)  

    def forward(self,x):
        x = self.sub_mean(x)
        first_out = self.head(x)
        out1 = self.unit1(first_out)
        out2 = self.unit2(out1)
        n_in1 = torch.cat([out1,out2],1)
        n_out1 = self.node1(n_in1)
    
        out3 = self.unit3(n_out1)
        out4 = self.unit4(out3)
        n_in2 = torch.cat([out3,out4],1)
        n_out2 = self.node2(n_in2)
        
        out5 = self.unit5(n_out2)
        out6 = self.unit6(out5)
        n_in3 = torch.cat([out5,out6,n_out1,n_out2],1)
        n_out3 = self.node3(n_in3)
        
        out7 = self.unit7(n_out3)
        out8 = self.unit8(out7)
        n_in4 = torch.cat([out7,out8],1)
        n_out4 = self.node4(n_in4)
        
        out9 = self.unit9(n_out4)
        out10 = self.unit10(out9)
        n_in5 = torch.cat([n_out4,out9,out10],1)
        n_out5 = self.node5(n_in5)
        
        out11 = self.unit11(n_out5)
        out12 = self.unit12(out11)
        n_in6 = torch.cat([out11,out12],1)
        n_out6 = self.node6(n_in6)
        
        out13 = self.unit13(n_out6)
        out14 = self.unit14(out13)
        n_in7 = torch.cat([out13,out14,n_out6,n_out5,n_out3,n_out1],1)
        n_out7 = self.node7(n_in7)
                
        res = self.tail_1(n_out7)
        out = self.tail_2(first_out)
        HR = torch.add(out,res) 
        HR = self.add_mean(HR) 
                             
        return HR
        
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