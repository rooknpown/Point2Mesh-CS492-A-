import numpy as np
import torch
import copy
import pickle

class Mesh():

    def __init__(self, file, hold_history=False, vertices = None, faces = None, device = 'cuda:0', gfmm = True):

        # print("Create Mesh")
        if file is None:
            # print("Mesh file does not exist at: " + str(file))
            return
        self.file = file
        self.device = device
        print(self.device)
    
        self.edges = None
        self.gemm_edges = None
        
        if vertices is not None and faces is not None:
            # print(len(vertices))
            # print(len(faces))
            self.vertices =  vertices.cpu().numpy()
            self.faces = faces.cpu().numpy()
            self.scale = 1.0
            self.translations = np.array([0,0,0])
        else:
            self.vertices, self.faces = self.load_obj(file)
            self.normalize()

        self.vs_init = copy.deepcopy(self.vertices)
        
        self.v_mask = np.ones(len(self.vertices))
        self.create_edges()

        
        self.temp_data = {}
        self.init_temp_data()
        if gfmm:
            self.gfmm = self.build_gfmm()
        else:
            self.gfmm = None

        self.vertices = torch.from_numpy(self.vertices).to(self.device)
        self.faces = torch.from_numpy(self.faces).to(self.device)

        self.area, self.normals = self.face_areas_normals(self.vertices, self.faces)


    def face_areas_normals(self, vertices, faces):
        if type(vertices) is not torch.Tensor:
            vertices = torch.from_numpy(vertices)
        if type(faces) is not torch.Tensor:
            faces = torch.from_numpy(faces)
        face_normals = torch.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                   vertices[faces[:, 2]] - vertices[faces[:, 1]])

        face_areas = torch.norm(face_normals, dim=1)
        face_normals = face_normals / face_areas[:, None]
        face_areas = 0.5 * face_areas
        face_areas = 0.5 * face_areas
        return face_areas, face_normals


    def print(self):
        print("Mesh info: \n" + "Vertices: " )
        print(self.vertices) 
        print(self.faces)
        print(self.edges)
        print(self.ecnt)
        print(self.nvsi)
        # print(self.ve



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
    
    def build_gfmm(self):
        edge_faces = self.build_ef()
        gfmm = []
        if type(self.faces) == torch.Tensor:
            faces = self.faces.cpu().numpy()
        else:
            faces = self.faces
        for face_id, face in enumerate(faces):
            neighbors = [face_id]
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                neighbors.extend(list(set(edge_faces[edge]) - set([face_id])))
            gfmm.append(neighbors)
        return torch.Tensor(gfmm).long().to(self.device)
    
    def build_ef(self):
        edge_faces = dict()
        if type(self.faces) == torch.Tensor:
            faces = self.faces.cpu().numpy()
        else:
            faces = self.faces
        for face_id, face in enumerate(faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(face_id)
        for k in edge_faces.keys():
            if len(edge_faces[k]) < 2:
                edge_faces[k].append(edge_faces[k][0])
        return edge_faces

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
        # print("edges_count")
        # print(self.ecnt)
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
        
        self.veidx = torch.from_numpy(np.concatenate(np.array(self.veidx, dtype= object)).ravel()).to(dev).long()
        self.nvsi = torch.Tensor(np.concatenate(np.array(self.nvsi, dtype=object).ravel())).to(dev).long()
        self.nvsin = torch.from_numpy(np.concatenate(np.array(self.nvsin, dtype=object)).ravel()).to(dev).long()
        ve_in = copy.deepcopy(self.ve)
        self.ve_in = torch.from_numpy(np.concatenate(np.array(ve_in, dtype=object)).ravel()).to(dev).long()
        self.max_nvs = max(self.nvs)
        self.nvs = torch.Tensor(self.nvs).to(dev).float()
        self.edge2key = edge2key

        # print(self.max_nvs)
    
    def init_temp_data(self):
        self.temp_data = {}
        self.temp_data['groups'] = []
        self.temp_data['gemm_edges'] = [self.gemm_edges.copy()]
        self.temp_data['occurrences'] = []
        self.temp_data['ecnt'] = [self.ecnt]


    def get_groups(self):
        return self.temp_data['groups'].pop()
    
    def get_occurrences(self):
        return self.temp_data['occurrences'].pop()

    def add_temp_data(self, groups, pool_mask):
        self.temp_data['groups'].append(groups.get_groups(pool_mask))
        self.temp_data['occurrences'].append(groups.get_sum())
        self.temp_data['gemm_edges'].append(self.gemm_edges.copy())
        self.temp_data['ecnt'].append(self.ecnt)

    def pop_temp_data(self):
        self.temp_data['gemm_edges'].pop()
        self.gemm_edges = self.temp_data['gemm_edges'][-1]
        self.temp_data['ecnt'].pop()
        self.ecnt = self.temp_data['ecnt'][-1]
    
    def clean(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve
        self.add_temp_data(groups, torch_mask)

    def update_vertices(self, vertices):
        self.vertices = vertices

    def deepcopy(self):
        new_mesh = Mesh(file=None)
        types = [np.ndarray, torch.Tensor,  dict, list, str, int, bool, float]
        for attr in self.__dir__():
            if attr == '__dict__':
                continue

            val = getattr(self, attr)
            if type(val) == types[0]:
                new_mesh.__setattr__(attr, val.copy())
            elif type(val) == types[1]:
                new_mesh.__setattr__(attr, val.clone())
            elif type(val) in types[2:4]:
                new_mesh.__setattr__(attr, pickle.loads(pickle.dumps(val, -1)))
            elif type(val) in types[4:]:
                new_mesh.__setattr__(attr, val)

        return new_mesh

class SubMesh():
    def __init__(self, base_mesh: Mesh, sub_num = 1, bfs_depth = 0):

        self.base_mesh = base_mesh
        self.subvertices, self.num_sub = self.set_shape(self.base_mesh.vertices, sub_num)
        # print("Sub prop")
        # print(self.subvertices)
        # print(self.num_sub)
        # print(bfs_depth)
        self.sub_mesh = []
        self.sub_mesh_idx = []
        self.init_vertices = []

        self.get_submesh(bfs_depth)



    def set_shape(self, vertices: torch.Tensor, sub_num):
        
        center = vertices.mean(dim=0)

        diff_vertices = vertices - center
        diff_vertices = diff_vertices + diff_vertices
        sub = torch.zeros(vertices.shape[0]).long().to('cuda:0')
        if sub_num >= 2:
            sub += 1*(diff_vertices[:,0]>0).long().to('cuda:0')
        if sub_num >= 4:
            sub += 2*(diff_vertices[:,1]>0).long().to('cuda:0')
        if sub_num >= 8:
            sub += 4*(diff_vertices[:,2]>0).long().to('cuda:0')

        num_sub = torch.max(sub).item() + 1

        return sub, num_sub

    def get_submesh(self, bfs_depth):
        
        subvertices2 = self.subvertices.clone()
        for i in range(self.num_sub):
            idx = torch.nonzero((self.subvertices == i), as_tuple=False)
            idx = idx.squeeze(1)
            # print(idx)
            if idx.size()[0] == 0:
                subvertices2[self.subvertices > i] -= 1
                continue
            idx = torch.sort(idx, dim = 0)[0]
            idx = self.bfs(idx, self.base_mesh.faces, bfs_depth).type(idx.dtype).clone().detach().to(idx.device)
            # print(idx)
            submesh, idx2 = self.create_submesh(idx)
            # print(idx2)
            self.sub_mesh_idx.append(idx2)
            self.sub_mesh.append(submesh)
            self.init_vertices.append(submesh.vertices.clone().detach())
        
        print(self.init_vertices)

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
            # print(self.submesh_e_idx)



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
        # print("Create Submesh")
        # print(idx)
        # print(mask.shape)
        mask[idx] = 1
        facemask = mask[mesh.faces].sum(dim = -1) > 0 
        faces2 = mesh.faces[facemask].clone()

        totvertices = faces2.view(-1)       

        mask2 = torch.zeros(len(mesh.vertices)).long().to(totvertices.device)
        mask2[totvertices] = 1

        idx2 = self.mask2index(mask2)
        # print(idx2 )
        vertices = mesh.vertices[idx2, :].clone()
        
        mask3 = torch.zeros(len(mesh.vertices))
        mask3[idx2] = 1
        cumsum = torch.cumsum(1 - mask3, dim = 0)
        faces2 -= cumsum[faces2].to(faces2.device).long()
        submesh = Mesh(file = mesh.file, vertices = vertices.detach(), faces = faces2.detach()
                        , device = mesh.device, gfmm = False)

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
    
    def __iter__(self):
        return iter(self.sub_mesh)

    def update_vertices(self, vertices, index):
        m = self.sub_mesh[index]
        m.update_vertices(vertices)
        # print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
        # self.base_mesh.vertices = self.base_mesh.vertices.to("cuda:0")
        # print(self.base_mesh.vertices)
        # print(vertices)
        # print(self.base_mesh.vertices[self.sub_mesh_idx[index], :])
        self.base_mesh.vertices[self.sub_mesh_idx[index], :] = vertices

    

    def build_base_mesh(self):

        new_vs = torch.zeros_like(self.base_mesh.vertices)
        new_vs_n = torch.zeros(self.base_mesh.vertices.shape[0], dtype=new_vs.dtype).to(new_vs.device)
        for i, m in enumerate(self.sub_mesh):
            new_vs[self.sub_mesh_idx[i], :] += m.vertices
            new_vs_n[self.sub_mesh_idx[i]] += 1
        new_vs = new_vs / new_vs_n[:, None]
        new_vs[new_vs_n == 0, :] = self.base_mesh.vertices[new_vs_n == 0, :]
        self.base_mesh.update_vertices(new_vs)