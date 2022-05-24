import torch
from mesh import Mesh
from pytorch3d.ops.knn import knn_gather, knn_points

class BeamGapLoss():
    def __init__(self, thres, device):
        self.thres = thres
        self.device = device
        self.points = None
        self.masks = None
        
        
    def update_params(self, mesh, dest_pc):
        points = []
        masks = []
        dest_pc.to(self.device)
        glob_mask = torch.zeros(mesh.base_mesh.vertices.shape[0])

        for i, j in enumerate(mesh):
            p, mask = self.projection(j, dest_pc, self.thres)
            p = p.to(dest_pc.device)
            mask = mask.to(dest_pc.device)
            points.append(p[:, :3])
            masks.append(mask)
            temp = torch.zeros(j.vertices.shape[0])
            if (mask != False).any():
                temp[j.faces[mask]] = 1
                glob_mask[mesh.sub_mesh_idx[i]] += temp
        self.points = points
        self.masks = masks
        
    def __call__(self, mesh, j):
        losses = self.points[j] - mesh.sub_mesh[j].vertices[mesh.sub_mesh[j].faces].mean(dim = 1)
        losses = ZeroNanGrad.apply(losses)
        losses = torch.norm(losses, dim = 1)[self.masks[j]]
        losses = losses.mean().float()
        return losses * 1e1

    def projection(self, j, dest_pc, thres):

        with torch.no_grad():
            device = torch.device('cpu')
            dest_pc = dest_pc.double()
            if isinstance(j, Mesh):
                mid_points = j.vertices[j.faces].mean(dim = 1)
                normals = j.normals
            else:
                mid_points = j[:,:3]
                normals = j[:, 3:]
            pk12 = knn_points(mid_points[:, :3].unsqueeze(0), dest_pc[:, :, :3], K=3).idx[0]
            pk21 = knn_points(dest_pc[:, :, :3], mid_points[:, :3].unsqueeze(0), K=3).idx[0]

            loop = pk21[pk12].view(pk12.shape[0], -1)
            knn_mask = (loop == torch.arange(0, pk12.shape[0], device=j.device)[:, None]).sum(dim=1) > 0
            mid_points = mid_points.to(device)
            dest_pc = dest_pc[0].to(device)

            normals = normals.to(device)[~ knn_mask, :]
            masked_mid_points = mid_points[~ knn_mask, :]
            displacement = masked_mid_points[:, None, :] - dest_pc[:, :3]

            torch.cuda.empty_cache()
            distance = displacement.norm(dim=-1)
            mask = (torch.abs(torch.sum((displacement / distance[:, :, None]) *
                                        normals[:, None, :], dim=-1)) > thres)
            if dest_pc.shape[-1] == 6:
                pc_normals = dest_pc[:, 3:]
                normals_correlation = torch.sum(normals[:, None, :] * pc_normals, dim=-1)
                mask = mask * (normals_correlation > 0)
            torch.cuda.empty_cache()


            distance[~ mask] += float('inf')
            min, argmin = distance.min(dim=-1)

            ppf_masked = dest_pc[argmin, :].clone()
            ppf_masked[min == float('inf'), :] = float('nan')
            ppf = torch.zeros(mid_points.shape[0], 6).\
                type(ppf_masked.dtype).to(ppf_masked.device)
            ppf[~ knn_mask, :dest_pc.shape[-1]] = ppf_masked
            ppf[knn_mask, :] = float('nan')


        return ppf.to(j.device), (ppf[:, 0] == ppf[:, 0]).to(device)






class ZeroNanGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, x):
        x[x != x] = 0
        return x