import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import ConstantPad2d


class MeshConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = [nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = (1, 5), bias = True)]
        self.conv = nn.ModuleList(self.conv)

    def forward(self, x, mesh):
        # print("MESH")
        # print(mesh)
        x = x.squeeze(-1)
        Gemm = self.create_gemm(x, mesh)
        # print(Gemm)
        x = self.conv[0](Gemm)
        return x      

    def create_gemm(self, x, mesh):
        padded_gem = []
        for i in mesh:
            gemm_inst = torch.tensor(i.gemm_edges, device = x.device).float().requires_grad_()
            gemm_inst = torch.cat((torch.arange(i.ecnt, device = x.device).float().unsqueeze(1), 
                                    gemm_inst), dim = 1)
            gemm_inst = F.pad(gemm_inst, (0, 0, 0, x.shape[2] - i.ecnt), "constant", 0)
            gemm_inst = gemm_inst.unsqueeze(0)
            padded_gem.append(gemm_inst)
        Gemm = torch.cat(padded_gem, dim = 0)

        # symmetric functions to handle order invariance
        
        Gshape = Gemm.shape
        # print("Gshape")
        # print(Gshape)

        pad = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad = True, device = x.device)

        x = torch.cat((pad, x), dim = 2)
        Gemm = Gemm + 1

        Gemm_flat = self.flatten_gemm(Gemm)
        # print("Gemm_flat")
        # print(Gemm_flat)

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

        f = torch.stack([f[:, :, :, 0], x1, x2, x3, x4], dim=3).float()

        return f
    
    def flatten_gemm(self, Gemm):
        (s1, s2, s3) = Gemm.shape
        s2 += 1
        fac = s2*torch.floor(torch.arange(s1*s2, device = Gemm.device).float()/s2).view(s1, s2)
        fac = fac.view(s1, s2, 1).repeat(1, 1, s3)


        Gemm = Gemm.float() + fac[:, 1:,:]
        Gemm = Gemm.view(-1).long()
        return Gemm



class MeshUnpool(nn.Module):
    def __init__(self, unpool_inst):
        super().__init__()
        self.unpool_inst = unpool_inst

    def forward(self, x, meshes):
        s1, s2, s3 = x.shape
        groups = self.get_pad_groups(meshes, s1, s3)
        occurrences = self.get_pad_occurrunces(meshes, s1)
        occurrences = occurrences.expand(groups.shape)

        groups = groups / occurrences
        groups = groups.to(x.device)
        for mesh in meshes:
            mesh.pop_temp_data()
        return torch.matmul(x, groups)
        
    def get_pad_groups(self, meshes, s1, s3):
        groups = []
        for mesh in meshes:
            group = mesh.get_groups()
            padrow = s3 - group.shape[0]
            padcol = self.unpool_inst - group.shape[1]
            if padrow != 0 or padcol != 0:
                padding = nn.ConstantPad2d((0, padcol, 0, pad_row), 0)
                group = padding(group)
            groups.append(group.clone().detach())
        
        groups = torch.cat(groups, dim = 0).view(s1, s3, -1)
        return groups

    def get_pad_occurrunces(self, meshes, s1):
        occurrences = []
        for mesh in meshes:
            occurrence = mesh.get_occurrences()
            pad = self.unpool_inst - occurrence.shape[0]
            if pad != 0:
                padding = nn.ConstantPad1d((0, padding), 1)
                occurrence = padding(occurrence)
            occurrences.append(occurrence)
        
        occurrences = torch.cat(occurrences, dim = 0).view(s1, 1, -1)

        return occurrences

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
        x = self.x[idx, : , : mesh.ecnt]
        x_sqsum = torch.sum(x ** 2, dim = 0)
        
        sorted, edge_ids = torch.sort(x_sqsum, descending = True)
        edge_ids = edge_ids.tolist()

        mask = np.ones(mesh.ecnt, dtype = np.bool)

        edge_groups = MeshUnion(mesh.ecnt, self.x.device)
        
        while mesh.ecnt > self.out_channel:
            edge_id = edge_ids.pop()
            if mask[edge_id]:
                self.pool_edge(mesh, edge_id, mask, edge_groups)
        mesh.clean(mask, edge_groups)
        x = edge_groups.rebuild(self.x[idx], mask, self.out_channel)
        self.new_x[idx] = x

    def pool_edge(self, mesh, edge_id, mask, edge_groups):
        for edge in mesh.gemm_edges[edge_id]:
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return

        if self.clean_side(mesh, edge_id, mask, edge_groups, 0) \
                and self.clean_side(mesh, edge_id, mask, edge_groups, 2) \
                and self.is_one_ring_valid(mesh, edge_id):
            self.pool_side(mesh, edge_id, mask, edge_groups, 0)
            self.pool_side(mesh, edge_id, mask, edge_groups, 2)
            mesh.merge_vertices(edge_id)
            mask[edge_id] = False
            mesh.ecnt -= 1


    def clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.ecnt <= self.out_channel:
            return False
        invalid_edges = self.get_invalids(mesh, edge_id, edge_groups, side)
        while len(invalid_edges) != 0 and mesh.ecnt > self.out_channel:
            self.remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.ecnt <= self.out_channel:
                return False
            for edge in mesh.gemm_edges[edge_id]:
                if edge == -1 or -1 in mesh.gemm_edges[edge]:
                    return False
            invalid_edges = self.get_invalids(mesh, edge_id, edge_groups, side)
        return True
    
    def is_one_ring_valid(self, mesh, edge_id):
        va = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))
        vb = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        shared = va & vb - set(mesh.edges[edge_id])
        if len(shared) == 2:
            return True
        return False
    
    def pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = self.get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        self.redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1],
                              mesh.sides[key_b, other_side_b + 1])
        edge_groups.union(key_b, key_a)
        edge_groups.union(edge_id, key_a)
        mask[key_b] = False
        mesh.remove_edge(key_b)
        mesh.ecnt -= 1
        return key_a
    
    def get_invalids(self, mesh, edge_id, edge_groups, side):
        info = self.get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b = info
        shared_items = self.get_shared_items(other_keys_a, other_keys_b)
        if len(shared_items) == 0:
            return []
        else:
            assert (len(shared_items) == 2)
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            self.redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            self.redirect_edges(mesh, edge_id, side + 1, update_key_b, update_side_b)
            self.redirect_edges(mesh, update_key_a, self.get_other_side(update_side_a), update_key_b,
                                      self.get_other_side(update_side_b))
            edge_groups.union(key_a, edge_id)
            edge_groups.union(key_b, edge_id)
            edge_groups.union(key_a, update_key_a)
            edge_groups.union(middle_edge, update_key_a)
            edge_groups.union(key_b, update_key_b)
            edge_groups.union(middle_edge, update_key_b)

            return [key_a, key_b, middle_edge]

    def redirect_edges(self, mesh, edge_a_key, side_a, edge_b_key, side_b):
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key
        mesh.sides[edge_a_key, side_a] = side_b
        mesh.sides[edge_b_key, side_b] = side_a

    def get_shared_items(self, list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items
    
    def get_other_side(self, side):
        return side + 1 - 2 * (side % 2)

    def get_face_info(self, mesh, edge_id, side):
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        other_side_a = (side_a - (side_a % 2) + 2) % 4
        other_side_b = (side_b - (side_b % 2) + 2) % 4
        other_keys_a = [mesh.gemm_edges[key_a, other_side_a], mesh.gemm_edges[key_a, other_side_a + 1]]
        other_keys_b = [mesh.gemm_edges[key_b, other_side_b], mesh.gemm_edges[key_b, other_side_b + 1]]
        return key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b

    def remove_triplete(self, mesh, mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            mask[edge_key] = False
        mesh.ecnt -= 3
        vertex = list(vertex)
        assert (len(vertex) == 1)
        mesh.remove_vertex(vertex[0])


class MeshUnion:
    def __init__(self, size, device):
        self.size = size
        self.groups = torch.eye(size, device=device)

    def union(self, src, dst):
        self.groups[dst, :] += self.groups[src, :]


    def rebuild(self, features, mask, dst_edges):
        mask = torch.from_numpy(mask)
        self.groups = torch.clamp(self.groups[mask, :], 0, 1).transpose_(1, 0)
        pad = features.shape[1] - self.groups.shape[0]
        if pad > 0:
            pad = ConstantPad2d((0, 0, 0, pad), 0)
            self.groups = pad(self.groups)

        x = torch.matmul(features.squeeze(-1), self.groups)
        occurrences = torch.sum(self.groups, 0).expand(x.shape)
        x = x / occurrences

        pad2 = dst_edges - x.shape[1]
        if pad2 > 0:
            pad2 = ConstantPad2d((0, pad2, 0, 0), 0)
            x = pad2(x)
        return x
    
    def get_groups(self, mask):
        self.groups = torch.clamp(self.groups, 0, 1)
        return self.groups[mask, :]

    def get_sum(self):
        return torch.sum(self.groups, 0)