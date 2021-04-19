import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from networks import blocks

# helpers functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

# lambda layer
class LambdaLayer(nn.Module):
    def __init__(
        self,
        dim,
        dim_k,
        n = None,
        r = None,
        heads = 4,
        dim_out = None,
        dim_u = 1):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.u = dim_u # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
        else:
            assert exists(n), 'You must specify the window size (n=h=w)'
            rel_lengths = 2 * n - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
            self.rel_pos = calc_rel_pos(n)

    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        k = k.softmax(dim=-1)

        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        if self.local_contexts:
            v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            λp = self.pos_conv(v)
            Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
        else:
            n, m = self.rel_pos.unbind(dim = -1)
            rel_pos_emb = self.rel_pos_emb[n, m]
            λp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b h v n', q, λp)

        Y = Yc + Yp
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return out


class LambdaBlock(nn.Module):
    def __init__(self, conv, in_channels, num_features):
        super(LambdaBlock, self).__init__()

        self.conv1 = conv(in_channels, num_features, 3)
        self.conv2 = conv(num_features, num_features, 1)
        self.conv3 = nn.ModuleList([LambdaLayer(dim=num_features,dim_out=num_features,r=23,dim_k=16, heads=4, dim_u=4)])
        self.conv3.append(nn.BatchNorm2d(num_features))
        self.conv3.append(nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(*self.conv3)
        self.conv4 = conv(num_features, num_features, 3)

    def forward(self, x):
        out = F.relu(self.conv1(x))+x
        out = self.conv2(out)
        out = self.conv3(out)
        out = F.relu(self.conv4(out), inplace=True)
        return out

class Lambdanet(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, upscale_factor, res_scale=1, conv=blocks.default_conv):
        super(Lambdanet, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        ##define network head
        net_head = [conv(in_channels, num_features, 3)]

        ##define network body
        net_body = nn.ModuleList()
        for i in range(20):
            net_body.append(LambdaBlock(conv, num_features, num_features))

        ##deine network tail
        net_tail = [
            conv(num_features, num_features, 3),
            blocks.Upsampler(conv, upscale_factor, num_features),
            conv(num_features, out_channels=3, kernel_size=3)
        ]  
        self.sub_mean = blocks.MeanShift(rgb_mean, rgb_std, sign=-1)
        self.head = nn.Sequential(*net_head)
        self.body = nn.Sequential(*net_body)
        # self.concat = conv(num_features*31, num_features, 1)
        self.tail = nn.Sequential(*net_tail)
        self.add_mean = blocks.MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, input):
        b, c, h, w = input.size()
        x = self.sub_mean(input)
        x = self.head(x)
        res = x

        # inter_out = []
        # for i in range(30):
        #     x = self.body[i](x)
        #     inter_out.append(x)

        res = self.body(x)
        res = torch.add(res, x)
        
        # inter_out.append(res)
        # res = torch.cat(inter_out, dim=1)
        # res = self.concat(res)

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

