import os
import numpy as np
import torch

class Mesh():

    def __init__(self, file, device):

        if not os.path.exists(file):
            print("Mesh file does not exist at: " + str(file))
            return

        self.device = device
    
        self.edges = None

        self.vertices, self.faces = self.load_obj(file)
        self.normalize()
        self.vertices = torch.from_numpy(self.vertices).to(self.device)
        self.faces = torch.from_numpy(self.faces).to(self.device)
        
        self.v_mask = np.ones(len(self.vertices))



    def load_obj(self, file):

        vertices = []
        faces = []
        f = open(file)
        for line in f:
            line = line.strip().split()

            if not line:
                continue
            # example line: v 0.716758 0.344326 -0.568914

            elif line[0] == 'v':
                coordList = []
                for x in line[1:4]:
                    coordList.append(float(x))
                vertices.append(coordList[:])

            # example line: f 1 4 2
            elif line[0] == 'f':
                coordList = []
                for x in line[1:4]:
                    coordList.append(int(x) - 1)
                faces.append(coordList[:])
        f.close()
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)
        return vertices, faces

    def normalize(self):
        maxCoord = []
        minCoord = []
        xyzScale = []
        self.translation = []
        for i in range(3):
            maxCoord.append(self.vertices[:,i].max())
            minCoord.append(self.vertices[:,i].min())
            xyzScale.append(maxCoord[i] - minCoord[i])
        self.scale = max(xyzScale)
        for i in range(3):
            self.translation.append((-maxCoord[i] - minCoord[i])/2/self.scale)


        self.vertices /= self.scale
        self.vertices += [self.translation]
        


class SubMesh():
    def __init__(self, base_mesh: Mesh, sub_num = 1, bfs_depth = 0):

        self.base_mesh = base_mesh
        self.subvertices, self.num_sub = self.set_shape(self.base_mesh.vertices, sub_num)

        self.sub_mesh = []
        self.sub_mesh_idx = []
        self.init_vertices = []

        self.get_submesh(bfs_depth)



    def set_shape(self, vertices: torch.Tensor, sub_num):
        
        center = vertices.mean(dim=0)

        diff_vertices = vertices - center
        diff_vertices = diff_vertices + diff_vertices
        sub = torch.zeros(vertices.shape[0]).long().to(diff_vertices.device)
        if sub_num >= 2:
            sub += 1*(diff_vertices[:,0]>0).long()
        if sub_num >= 4:
            sub += 2*(diff_vertices[:,1]>0).long()
        if sub_num >= 8:
            sub += 4*(diff_vertices[:,2]>0).long()

        num_sub = torch.max(sub).item() + 1

        return sub, num_sub

    def get_submesh(self, bfs_depth):
        
        for i in range(self.num_sub):
            idx = torch.nonzero((self.subvertices == i), as_tuple=False)
            print(idx.shape)
            idx = idx.squeeze(1)
            if idx.size()[0] == 0:
                continue
            idx = torch.sort(self.subvertices, dim = 0)[0]
            idx = self.bfs(idx, self.base_mesh.faces, bfs_depth).type(idx.dtype).clone().detach().to(idx.device)
            self.create_submesh(idx)

    def bfs(self, idxs, faces, bfs_depth):
        if bfs_depth <= 0:
            return idxs
        
        idxs = idxs.tolist()
        faces = faces.tolist()

        marked = []
        for vertice in idxs:
            marked.append((vertice, 0))
        visited = idxs
        while len(marked) >0:
            idx, depth = marked.pop()
            for f in faces:
                if idx in f:
                    for j in f:
                        if j not in visited:
                            if depth + 1 <= bfs_depth:
                                marked.put((j, depth + 1))
                            visited.append(j)
        visited = sorted(visited)
        return visited

    def create_submesh(self, idx):

        mesh = self.base_mesh
        mask = torch.zeros(len(mesh.vertices))
        mask[idx] = 1
        facemask = mask[mesh.faces].sum(dim = -1) > 0 
        newfaces = mesh.faces[facemask].clone()

        totvertices = newfaces.reshape(-1)       

        mask2 = torch.zeros(len(mesh.vertices)).long().to(totvertices.device)
        mask2[totvertices] = 1

        idx2 = self.mask2index(mask2)
        vertices = mesh.vertices[idx2, :].clone()
        
        mask3 = torch.zeros(len(mesh.vertices))
        mask3[idx2] = 1
        cumsum = torch.cumsum(1 - mask3, dim = 0)
        
        
    def mask2index(self, mask):
        idx = []
        for i, j in enumerate(mask):
            if j == 1:
                idx.append(i)
        return torch.tensor(idx).type(torch.long)
