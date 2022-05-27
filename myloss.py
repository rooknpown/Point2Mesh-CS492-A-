
import torch
import pytorch3d
from mesh import Mesh 
from pytorch3d.ops.knn import knn_points, knn_gather

def bidir_chamfer_loss(xcoord, ycoord, xnormal, ynormal):
    
    xlength = torch.tensor([xcoord.shape[1]], device = xcoord.device)
    ylength = torch.tensor([ycoord.shape[1]], device = ycoord.device)
    

    xknn = knn_points(xcoord, ycoord, xlength, ylength, 1)
    yknn = knn_points(ycoord, xcoord, ylength, xlength, 1)

    
    cham_x = torch.sqrt(xknn.dists[..., 0])
    cham_y = torch.sqrt(yknn.dists[..., 0])

    xnknn = knn_gather(ynormal, xknn.idx, ylength)[..., 0, :]
    ynknn = knn_gather(xnormal, yknn.idx, xlength)[..., 0, :]


    cossim = torch.nn.CosineSimilarity(dim = 2, eps = 1e-6)
    cham_nx = -1 * abs(cossim(xnormal, xnknn))
    cham_ny = -1 * abs(cossim(ynormal, ynknn))
   

    chamsumx = cham_x.sum()/xlength [0]
    chamsumy = cham_y.sum()/ylength [0]
    chamsumnx = cham_nx.sum()/xlength [0]
    chamsumny = cham_ny.sum()/ylength [0]

    chamfer_dist = chamsumx + chamsumy
    chamfer_normals = chamsumnx + chamsumny

    return chamfer_dist, chamfer_normals

    

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

        for i in range(len(mesh.sub_mesh)):
            p, mask = self.projection(mesh.sub_mesh[i], dest_pc, self.thres)
            p = p.to(dest_pc.device)
            mask = mask.to(dest_pc.device)
            points.append(p[:, :3])
            masks.append(mask)
            temp = torch.zeros(mesh.sub_mesh[i].vertices.shape[0])
            if (mask != False).any():
                temp[mesh.sub_mesh[i].faces[mask]] = 1
                glob_mask[mesh.sub_mesh_idx[i]] += temp
        self.points = points
        self.masks = masks
        
    def __call__(self, mesh, j):
        losses = self.points[j] - mesh.sub_mesh[j].vertices[mesh.sub_mesh[j].faces].mean(dim = 1)
        losses = torch.norm(SimpleGrad.apply(losses), dim = 1)[self.masks[j]]
        losses = losses.mean().float() * 1e1
        return losses 

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

            facepoints = mid_points[:, :3].unsqueeze(0)
            collpoints = dest_pc[:, :, :3]

            kp12 = knn_points(facepoints, collpoints, K=3).idx[0]
            kp21 = knn_points(collpoints, facepoints, K=3).idx[0]

            masklen = kp12.shape[0]
            loop = kp21[kp12].view(masklen, -1)
            knn_mask = (loop == torch.arange(0, masklen, device=j.device)[:, None]).sum(dim=1) > 0

            mid_points = mid_points.to(device)
            dest_pc = dest_pc[0].to(device)

            normals = normals.to(device)[~ knn_mask, :]
            delta = mid_points[~ knn_mask, :][:, None, :] - dest_pc[:, :3]
            dest_shape = dest_pc.shape[-1]

            torch.cuda.empty_cache()
            distance = delta.norm(dim=-1)
            mask = (torch.abs(torch.sum((delta / distance[:, :, None]) *
                                        normals[:, None, :], dim=-1)) > thres)
            if dest_shape == 6:
                pc_normals = dest_pc[:, 3:]
                normals_correlation = torch.sum(normals[:, None, :] * pc_normals, dim=-1)
                mask = mask * (normals_correlation > 0)
            torch.cuda.empty_cache()


            distance[~ mask] += float('inf')
            min, argmin = distance.min(dim=-1)

            ppf_masked = dest_pc[argmin, :].clone()
            ppf_masked[min == float('inf'), :] = float('nan')
            ppf = torch.zeros(mid_points.shape[0], 6).type(torch.float64).to(ppf_masked.device)
            ppf[~ knn_mask, :dest_shape] = ppf_masked
            ppf[knn_mask, :] = float('nan')



        return ppf.to(j.device), (ppf[:, 0] == ppf[:, 0]).to(device)




class SimpleGrad(torch.autograd.Function):
    @staticmethod
    def forward(t, x):
        return x

    @staticmethod
    def backward(t, x):
        x[x != x] = 0
        return x