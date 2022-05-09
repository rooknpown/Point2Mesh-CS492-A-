import torch
import torch.nn as nn
from model_utils import init_weight
import torch.nn.functional as F

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
        Gemm = self.create_gemm(x, mesh)
        x = self.conv(Gemm)
        return x      

    def create_gemm(self, x, mesh):
        padded_gem = []
        for i in mesh:
            gemm_inst = torch.tensor(i.gemm_edges, device = x.device).float.requires_grad_()
            gemm_inst = torch.cat((torch.arange(m.edges_count, device = x.device).float().unsqueeze(1), 
                                    gemm_inst), dim = 1)
            gemm_inst = F.pad(gemm_inst, (0, 0, 0, x.shape[2] - m.edges_count), "constant", 0)
            gemm_inst = gemm_inst.unsqueeze(0)
            padded_gem.append(gemm_inst)
        Gemm = torch.cat(padded_gem, dim = 0)

        # symmetric functions to handle order invariance
        
        Gshape = Gemm.shape

        pad = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad = True, device = x.device)

        x = torch.cat((pad, x), dim = 2)
        Gemm = Gemm + 1

        Gemm_flat = self.flatten_gemm(Gemm)

        out_dim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(out_dim[0] * out_dim[2], out_dim[1])

        f = torch.index_select(x, dim=0, index=Gemm_flat)
        f = f.view(Gshape[0], Gshape[1], Gshape[2], -1)
        f = f.permute(0, 3, 1, 2)

        x1 = f[:, :, :, 1] + f[:, :, :, 3]
        x2 = f[:, :, :, 2] + f[:, :, :, 4]

        x3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
        x4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])

        f = torch.stack([f[:, :, :, 0], x1, x2, x3, x4], dim=3)

        return f
    
    def flatten_gemm(self, Gemm):
        (s1, s2, s3) = Gemm.shape
        s2 += 1
        fac = s2*torch.floor(torcj.arange(s1*s2, device = Gemm.device).float()/s2).view(s1, s2)
        fac = fac.view(s1, s2, 1).repeat(1, 1, s3)

        Gemm = Gemm.float() + fac[:, 1:,:]
        Gemm.view(-1).long()
        return Gemm

class MeshPool(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.x = None
        self.new_x = None
        self.meshes = None

    def forward(self, x, meshes):
        self.new_x = [[] * len(meshes)]
        self.x = x
        self.meshes = meshes

        for i in range(len(meshes)):
            self.pool(i)
        out = torch.cat(self.new_x).view(len(meshes), -1, self.out_channel)
        return out

    def pool(self, idx):
        mesh = self.meshes[idx]
        x = self.x[idx, : , : mesh.edges_count]
        x_sqsum = torch.sum(x ** 2, dim = 0)
        
        sorted, edge_ids = torch.sort(x_sqsum, descending = True)
        edge_ids = edge_ids.tolist()

        mask = np.ones(mesh.edges_count, dtype = np.bool)
        

        

class MeshUnpool(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass


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
        self.unpool = MeshUnpool(unpool_inst)
        self.leaky = leaky
        self.res_blocks = res_blocks


    def forward(self, x, meshes, nopool):
        x = self.conv1(x, meshes).squeeze(3)
        x = self.unpool(x, meshes)
        if self.transfer:
            x = torch.cat((x, nopool), 1)
        x = self.conv2(x, meshes)
        x = F.leaky_relu(x, self.leaky)
        x = self.bn(x)

        x2 = x
        for i in range(self.res_blocks):
            x2 = conv3(x, meshes)
            x2 = F.leaky_relu(x2, self.leaky)
            x2 = self.bn(x2)
            x2 = x2 + x1
            x1 = x2
        x2 = x2.squeeze(3)

        return x2
        

class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel, res_blocks, pool_inst, leaky):
        super().__init__()
        self.conv1 = MeshConv(in_channel, out_channel)
        self.conv2 = MeshConv(out_channel, out_channel)
        self.bn = nn.InstanceNorm2d(out_channel)
        self.pool = MeshPool(pool_inst)
        self.leaky = leaky
        self.res_blocks = res_blocks
        
    def forward(self, x, meshes):
        x = self.conv1(x, meshes)
        x = F.leaky_relu(x, self.leaky)
        x = self.bn(x)

        x2 = x
        for i in range(self.res_blocks):
            x2 = conv2(x1, meshes)
            x2 = F.leaky_relu(x2, self.leaky)
            x2 = self.bn(x)
            x2 = x2 + x
            x = x2
        x2 = x2.squeeze(3)
        nopool = x2
        x2 = self.pool(x2, meshes)

        return x2, nopool
        



