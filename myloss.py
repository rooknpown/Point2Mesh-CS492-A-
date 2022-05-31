
import torch
import pytorch3d
from authors.mesh import Mesh 
from authors.utils import SimpleGrad
from pytorch3d.ops.knn import knn_points, knn_gather

def bidir_chamfer_loss(xcoord, ycoord, xnormal, ynormal):
    
    xlength = torch.tensor([xcoord.shape[1]], device = xcoord.device)
    ylength = torch.tensor([ycoord.shape[1]], device = ycoord.device)
    

    xknn = knn_points(xcoord, ycoord, xlength, ylength, 1)
    yknn = knn_points(ycoord, xcoord, ylength, xlength, 1)

    
    cham_x = torch.sqrt(xknn.dists[..., 0])
    cham_y = torch.sqrt(yknn.dists[..., 0])

    xnknn = knn_gather(ynormal, xknn.idx, ylength)
    ynknn = knn_gather(xnormal, yknn.idx, xlength)
    
    xnknn = xnknn[..., 0, :]
    ynknn = ynknn[..., 0, :]


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
        self.points = []
        self.masks = []
        
        
    def update_params(self, mesh, dest_pc):

        self.points.clear()
        self.masks.clear()

        glob_mask = torch.zeros(mesh.base_mesh.vertices.shape[0])
        dest_pc.to(self.device)
        

        for i in range(len(mesh.sub_mesh)):
            p, mask = mesh.sub_mesh[i].projection(dest_pc, self.thres)
            p = p.to(dest_pc.device)
            mask = mask.to(dest_pc.device)
            masks.append(mask)
            psliced = p[:, :3]
            points.append(psliced)
            cnst = torch.zeros(mesh.sub_mesh[i].vertices.shape[0])
            if (mask != False).any():
                cnst[mesh.sub_mesh[i].faces[mask]] = 1
                glob_mask[mesh.sub_mesh_idx[i]] += cnst
        self.points = points
        self.masks = masks
        
    def __call__(self, mesh, j):
        losses = self.points[j] - mesh.sub_mesh[j].vertices[mesh.sub_mesh[j].faces].mean(dim = 1)
        losses = torch.norm(SimpleGrad.apply(losses), dim = 1)[self.masks[j]]
        losses = losses.mean().float() * 1e1
        return losses 