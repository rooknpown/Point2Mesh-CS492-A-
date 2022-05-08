import torch
import torch.nn as nn
from model_utils import init_weight

## Self-prior



class PriorNet(nn.Module):
    def __init__(self, sub_mesh, in_channel, convs, pool, res_blocks, 
                leaky, transfer, init_weights, init_vertices):
        super().__init__()
        convs = list(convs)
        templist = [i for i in range(len(convs), 0,-1)]

        down_convs = [in_channel] + convs
        up_convs = down_convs[:]
        up_convs.reverse()
        pool = [len(convs)] + [pool]

        self.encdec = MeshEncoderDecoder(down_convs = down_convs, up_convs = up_convs,
                                        pool = pool, res_blocks = res_blocks, leaky = leaky,
                                        transfer = transfer)
        self.end_conv = MeshConv(in_channel, in_channel)
        init_weight(self, init_weights, 'normal_')
        init_weight(self.end_conv, 1e-8, 'uniform_')

        self.init_vertices = init_vertices

        

    def forward(self):
        pass


class MeshEncoderDecoder(nn.Module):
    def __init__(self, down_convs, up_convs, pool, res_blocks, leaky, transfer):
        super().__init__()
        self.encoder = MeshEncoder(down_convs = down_convs, pool = pool, 
                                    res_blocks = res_blocks, leaky = leaky)
        unpool = pool[:]
        unpool.reverse()
        self.decoder = MeshDecoder(up_convs = up_convs, unpool = unpool, 
                                    res_blocks = res_blocks, transfer = transfer)
        self.batch_norm  = nn.InstanceNorm2d(up_convs[-1])
            
    def forward(self, x, meshes):
        x, nopool = self.encoder(x, mesehs)
        x = self.decoder(x, meshes, nopool)
        x = self.batch_norm(x.unsqueeze(-1))
        return x


class MeshEncoder(nn.Module):
    def __init__(self ,down_convs, pool, res_blocks, leaky):
        super().__init__()
        self.convs = []
        for i in range(len(down_convs)-1):
            if i < len(pool) - 1:
                pool_inst = pool[i+1] 
            else:
                pool_inst = 0
            self.convs.append(DownConv(down_convs[i], down_convs[i+1], res_blocks = res_blocks,
            pool_inst = pool_inst, leaky = leaky))
        self.convs = nn.ModuleList(self.convs)
        init_weight(self, 0, "conv_xavier_normal_")


    def forward(self, in_x, meshes):
        out = []
        x = in_x
        for conv in self.convs:
            x, nopool = conv(x, meshes)
            out.append(nopool)
        return x, out
        
class MeshDecoder(nn.Module):
    def __init__(self, up_convs, unpool, res_blocks, transfer):
        super().__init__()
        self.convs = []
        for i in range(len(up_convs) - 2):
            if len(unpool) > idx:
                unpool_inst = unpool[i]
            else:
                unpool_inst = 0
            self.up_convs.append(UpConv(up_convs[i], up_convs[i+1], res_blocks = res_blocks,
                                unpool = unpool_inst, transfer = transfer, leaky = leaky))
            
        self.final_conv = UpConv(convs[-2], convs[-1], res_blocks = res_blocks,
                                unpool = False, transfer = False, leaky = leaky)
        self.convs = nn.ModuleList(self.convs)
        init_weight(self, 0, "conv_xavier_normal_")
        
    def forward(self, x, meshes, encoder_out):
        for i, conv in enumerate(self.convs):
            nopool = None
            if encoder_out is not None:
                nopool = encoder_out[-(i+2)]
            x = up_conv(x, meshes, nopool)
        x = self.final_conv(x, meshes)
        return x
        

class MeshConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = (1, k), bias = True)
    def forward(self, x, mesh):
        x = x.squeeze(-1)
class MeshPool(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass
class MeshUnpool(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass
class UpConv(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass
class DownConv(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass
class MeshUnion():
    def __init__(self):
        pass



