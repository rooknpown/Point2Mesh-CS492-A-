import torch
import torch.nn as nn
from authors.utils import init_weight, make3, array_times
import torch.nn.functional as F
from authors.meshcnn import MeshPool, MeshConv, MeshUnpool
from operator import attrgetter
import copy
import numpy as np

## Self-prior



class PriorNet(nn.Module):
    def __init__(self, sub_mesh, in_channel, convs, pool, res_blocks, 
                leaky, transfer, init_weights, init_vertices, disable_net):
        super().__init__()
        self.end_conv = MeshConv(in_channel, in_channel)
        self.disable_net = disable_net
        convs = list(convs)
        templist = [i for i in range(len(convs), 0,-1)]
        self.factor_pools = pool
        down_convs = [in_channel] + convs
        up_convs = down_convs[:]
        up_convs.reverse()
        pool = [len(convs)] + templist[1:]
        # print("Pool: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(pool)

        

        self.init_vertices = init_vertices
        self.encdec = MeshEncoderDecoder(down_convs = down_convs, up_convs = up_convs,
                                        pool = pool, res_blocks = res_blocks, leaky = leaky,
                                        transfer = transfer)
        
        
        
        # print(self.init_vertices)

        # rearrange pooling
        self.pools = [i for i in self.modules() if isinstance(i, MeshPool)]
        self.unpools = [i for i in self.modules() if isinstance(i, MeshUnpool)]
            # if isinstance(i, MeshPool):
            #     self.pools.append(i)
            # if isinstance(i, MeshUnpool):
            #     self.unpools.append(i) lambda x: x.out_channel

        self.pools = sorted(self.pools, key = attrgetter('out_channel'), reverse = True)
        self.unpools = sorted(self.unpools, key = attrgetter('unpool_inst'))
        init_weight(self, init_weights, 'normal_')
        self.init_sub_vertices = nn.ParameterList([torch.nn.Parameter(i) for i in sub_mesh.init_vertices])
        init_weight(self.end_conv, 1e-8, 'uniform_')  
        for i in self.init_sub_vertices:
            i.requires_grad = False



    def forward(self, x, sub_mesh):
        for i, p in enumerate(sub_mesh):
            self.init_vertices = self.init_sub_vertices[i]
            # print("pools")
            edgenum = p.ecnt
            temp_pools = [int(edgenum - i) for i in make3(array_times( edgenum, self.factor_pools))] 
            # print(edgenum)
            # print(self.pools)
            # print(temp_pools)

            for j in range(len(self.pools)):
                self.pools[j].out_channel = temp_pools[j]

            temp_pools.insert(0, edgenum)
            temp_pools.pop()
            temp_pools.reverse()
            for j in range(len(self.unpools)):
                self.unpools[j].unpool_inst = temp_pools[j]
 
            relevant_edges = x[:, :, sub_mesh.submesh_e_idx[i]]
            new_mesh = [p.deepcopy()]
            x2, y = self.encdec(relevant_edges, new_mesh)

            # print("AAAAA")
            # print(x)
            x2 = self.end_conv(x2.squeeze(-1), new_mesh)
            x2 = x2.squeeze(-1).unsqueeze(0)
            # print(x.unsqueeze(0))

            verts = self.build_verts(x2, p, 1)
            if self.disable_net:
                verts = self.build_verts(relevant_edges.unsqueeze(0), p, 1).requires_grad_()
                yield verts.double()
            # print(verts.float().shape)
            # print(self.init_vertices.expand_as(verts).shape)
            else:
                yield verts.float() + self.init_vertices.expand_as(verts).to(verts.device)
    

    def build_verts(self, x, mesh, l):
        x = x.reshape(l, 2, 3, -1)
        vs2sum = torch.zeros([l, len(mesh.vs_init), mesh.max_nvs, 3] ,dtype = x.dtype, device = x.device)
        # print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
        x = x[:, mesh.veidx, :, mesh.ve_in].transpose(0, 1)
        vs2sum[:, mesh.nvsi, mesh.nvsin, :] = x
        vs_sum = torch.sum(vs2sum, dim=2)
        nvs = mesh.nvs.to("cuda:0")
        vs = vs_sum / nvs[None, :, None]
        return vs

class MeshEncoderDecoder(nn.Module):
    def __init__(self, down_convs, up_convs, pool, res_blocks, leaky, transfer):
        super().__init__()
        # print("Encoder")
        # print(pool)
        self.encoder = MeshEncoder(down_convs = down_convs, pool = pool, 
                                    res_blocks = res_blocks, leaky = leaky)
        unpool = pool[:-1].copy()
        unpool.reverse()
        self.decoder = MeshDecoder(up_convs = up_convs, unpool = unpool, 
                                    res_blocks = res_blocks, transfer = transfer, leaky = leaky)
        self.batch_norm  = nn.InstanceNorm2d(up_convs[-1])
            
    def forward(self, x, meshes):
        x, nopool = self.encoder(x, meshes)
        x = self.decoder(x, meshes, nopool)
        x = self.batch_norm(x.unsqueeze(-1))
        return x, None


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
    def __init__(self, up_convs, unpool, res_blocks, transfer, leaky):
        super().__init__()
        self.convs = []
        for i in range(len(up_convs) - 2):
            if len(unpool) > i:
                unpool_inst = unpool[i]
            else:
                unpool_inst = 0
            self.convs.append(UpConv(up_convs[i], up_convs[i+1], res_blocks = res_blocks,
                                unpool_inst = unpool_inst, transfer = transfer, leaky = leaky))
            
        self.final_conv = UpConv(up_convs[-2], up_convs[-1], res_blocks = res_blocks,
                                unpool_inst = False, transfer = False, leaky = leaky)
        self.convs = nn.ModuleList(self.convs)
        init_weight(self, 0, "conv_xavier_normal_")
        
    def forward(self, x, meshes, encoder_out):
        for i, conv in enumerate(self.convs):
            nopool = None
            if encoder_out is not None:
                nopool = encoder_out[-(i+2)]
            x = conv(x, meshes, nopool)
        x = self.final_conv(x, meshes, None)
        return x
        



class UpConv(nn.Module):
    def __init__(self,  in_channel, out_channel, res_blocks, unpool_inst, transfer, leaky):
        super().__init__()
        self.conv1 = MeshConv(in_channel, out_channel)
        self.transfer = transfer
        if transfer:
            self.conv2 = MeshConv(2*out_channel, out_channel)
        else:
            self.conv2 = MeshConv(out_channel, out_channel)
        
        self.conv3 = MeshConv(out_channel, out_channel)
        self.bn = nn.InstanceNorm2d(out_channel)
        self.unpool = None
        if unpool_inst:
            self.unpool = MeshUnpool(unpool_inst)
        self.leaky = leaky
        self.res_blocks = res_blocks


    def forward(self, x, meshes, nopool):
        x = self.conv1(x, meshes).squeeze(3)
        if self.unpool:
            x = self.unpool(x, meshes)
        if self.transfer:
            x = torch.cat((x, nopool), 1)
        x = self.conv2(x, meshes)
        x = F.leaky_relu(x, self.leaky)
        x = self.bn(x)

        x2 = x
        for i in range(self.res_blocks):
            x2 = self.conv3(x, meshes)
            x2 = F.leaky_relu(x2, self.leaky)
            x2 = self.bn(x2)
            x2 = x2 + x
            x1 = x2


        return x2.squeeze(3)
        

class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel, res_blocks, pool_inst, leaky):
        super().__init__()
        self.conv1 = MeshConv(in_channel, out_channel)
        self.conv2 = MeshConv(out_channel, out_channel)
        self.bn = nn.InstanceNorm2d(out_channel)
        self.pool = None
        if pool_inst:
            self.pool = MeshPool(pool_inst)
        self.leaky = leaky
        self.res_blocks = res_blocks
        
    def forward(self, x, meshes):
        # print("MESHes downconv")
        # print(meshes)
        nopool = None
        x = self.conv1(x, meshes)
        x = F.leaky_relu(x, self.leaky)
        x = self.bn(x)

        x2 = x
        for i in range(self.res_blocks):
            x2 = self.conv2(x, meshes)
            x2 = F.leaky_relu(x2, self.leaky)
            x2 = self.bn(x)
            x2 = x2 + x
            x = x2
        
        x2 = x2.squeeze(3)
        
        if self.pool is not None:
            nopool = x2
            x2 = self.pool(x2, meshes)

        return x2, nopool
        

