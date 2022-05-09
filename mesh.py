import os
import numpy as np
import torch
import copy

class Mesh():

    def __init__(self, file, vertices = None, faces = None, device = 'cpu'):

        if not os.path.exists(file):
            print("Mesh file does not exist at: " + str(file))
            return
        self.file = file
        self.device = device
    
        self.edges = None
        self.gemm_edges = None
        
        if vertices is not None and faces is not None:
            self.vertices =  vertices.cpu().numpy()
            self.faces = faces.cpu().numpy()
            self.scale = 1.0
            self.translations = np.array([0,0,0])
        else:
            self.vertices, self.faces = self.load_obj(file)
            self.normalize()

        self.vertices = torch.from_numpy(self.vertices).to(self.device)
        self.faces = torch.from_numpy(self.faces).to(self.device)
        
        self.v_mask = np.ones(len(self.vertices))
        self.create_edges()

        self.temp_data = {}
        self.init_temp_data()



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
    

    def create_edges(self):
        self.ve = [[] for _ in self.vertices]
        self.veidx = [[] for _ in self.vertices]


        edges = []
        edge_nb = []
        sides = []
        edge2key = {}
        ecnt = 0
        count = []
        for faceid, face in enumerate(self.faces):
            face_edges = []
            for i in range(3):
                edge1 = (face[i], face[(i+1) % 3])
                face_edges.append(edge1)
            for idx, edge in enumerate(face_edges):
                edge = tuple(sorted(list(edge)))
                face_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = ecnt
                    edges.append(list(edge))
                    edge_nb.append([-1] * 4)
                    sides.append([-1] * 4)
                    self.ve[edge[0]].append(ecnt)
                    self.ve[edge[1]].append(ecnt)
                    self.veidx[edge[0]].append(0)
                    self.veidx[edge[1]].append(1)
                    count.append(0)
                    ecnt += 1

            for idx, edge in enumerate(face_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][count[edge_key]] = edge2key[face_edges[(idx + 1) % 3]]
                edge_nb[edge_key][count[edge_key] + 1] = edge2key[face_edges[(idx + 2) % 3]]
                count[edge_key] += 2

            for idx, edge in enumerate(face_edges):
                edge_key = edge2key[edge]
                sides[edge_key][count[edge_key] - 2] = count[edge2key[face_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][count[edge_key] - 1] = count[edge2key[face_edges[(idx + 2) % 3]]] - 2
        self.edges = np.array(edges, dtype = np.int64)
        self.sides = np.array(sides, dtype = np.int64)
        self.ecnt = ecnt
        
        # Gemm edges for mesh convolution
        self.gemm_edges = np.array(edge_nb, dtype = np.int64)
        # Loss DSs

        self.nvs = []
        self.nvsi = []
        self.nvsin = []

        for i, j in enumerate(self.ve):
            self.nvs.append(len(j))
            self.nvsi.append(len(j)* [i])
            self.nvsin.append(list(range(len(j))))
        
        dev = self.device
        
        self.veidx = torch.from_numpy(np.concatenate(np.array(self.veidx, dtype=object)).ravel()).to(dev)
        self.nvsi = torch.Tensor(np.concatenate(np.array(self.nvsi, dtype=object).ravel())).to(dev)
        self.nvsin = torch.from_numpy(np.concatenate(np.array(self.nvsin, dtype=object)).ravel()).to(dev)
        ve_in = copy.deepcopy(self.ve)
        self.ve_in = torch.from_numpy(np.concatenate(np.array(ve_in, dtype=object)).ravel()).to(dev)
        self.max_nvs = max(self.nvs)
        self.nvs = torch.Tensor(self.nvs).to(dev).float()
        self.edge2key = edge2key
    
    def init_temp_data(self):
        self.temp_data['groups'] = []
        self.temp_data['gemm_edges'] = [self.gemm_edges.copy()]
        self.temp_data['occurrences'] = []
        self.temp_data['ecnt'] = [self.ecnt]


    def get_groups(self):
        return self.temp_data['groups'].pop()
    
    def get_ocurrences(self):
        return self.temp_data['occurrences'].pop()

    def add_temp_data(self, groups, pool_mask):
        self.temp_data['groups'].append(groups.get_groups(pool_mask))
        self.temp_data['occurrences'].append(groups.get_ocurrences())
        self.temp_data['gemm_edges'].append(self.gemm_edges.copy())
        self.temp_data['ecnt'].append(self.ecnt)

    def pop_temp_data(self):
        self.temp_data['gemm_edges'].pop()
        self.gemm_edges = self.temp_data['gemm_edges'][-1]
        self.temp_data['ecnt'].pop()
        self.ecnt = self.temp_data['ecnt'][-1]

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
        
        subvertices2 = self.subvertices.clone()
        for i in range(self.num_sub):
            idx = torch.nonzero((self.subvertices == i), as_tuple=False)
            print(idx.shape)
            idx = idx.squeeze(1)
            if idx.size()[0] == 0:
                subvertices2[self.subvertices > i] -= 1
                continue
            idx = torch.sort(self.subvertices, dim = 0)[0]
            idx = self.bfs(idx, self.base_mesh.faces, bfs_depth).type(idx.dtype).clone().detach().to(idx.device)
            submesh, idx2 = self.create_submesh(idx)
            self.sub_mesh_idx.append(idx2)
            self.sub_mesh.append(submesh)
            self.init_vertices.append(submesh.vertices.clone().detach())
        
        self.subvertices = subvertices2
        self.num_sub = torch.max(self.subvertices).item() + 1

        bme = self.base_mesh.edges
        vertice_edge_dict = self.create_dict(bme)
        self.submesh_e_idx = []
        for i in range(self.num_sub):
            mask = torch.zeros(len(bme))
            for face in self.sub_mesh[i].faces:
                face = self.sub_mesh_idx[i][face].to(face.device)
                for j in range(3):
                    edge = tuple(sorted([face[j].item(), face[(j+1) % 3].item()]))
                    mask[vertice_edge_dict[edge]] = 1
            self.submesh_e_idx.append(self.mask2index(mask))



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
        faces2 = mesh.faces[facemask].clone()

        totvertices = faces2.reshape(-1)       

        mask2 = torch.zeros(len(mesh.vertices)).long().to(totvertices.device)
        mask2[totvertices] = 1

        idx2 = self.mask2index(mask2)
        vertices = mesh.vertices[idx2, :].clone()
        
        mask3 = torch.zeros(len(mesh.vertices))
        mask3[idx2] = 1
        cumsum = torch.cumsum(1 - mask3, dim = 0)
        faces2 -= cumsum[faces2].to(faces2.device).long()
        submesh = Mesh(file = mesh.file, vertices = vertices.detach(), faces = faces2.detach()
                        , device = mesh.device)

        return submesh, idx2

        
    def mask2index(self, mask):
        idx = []
        for i, j in enumerate(mask):
            if j == 1:
                idx.append(i)
        return torch.tensor(idx).type(torch.long)

    def create_dict(self, edges):
        dic = {}
        for i, j in enumerate(edges):
            t = tuple(sorted(j))
            dic[t] = i
        return dic
