import torch
import torch.nn as nn
from torch.nn import init
import sys
import os
from authors.mesh import Mesh
from myutils import export
import uuid
import glob


def init_weight(model, init_weights, init_type):
    def init_w(model):
        classname = model.__class__.__name__
        if hasattr(model, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal_':
                init.normal_(model.weight.data, 0.0, init_weights)
            elif init_type == 'uniform_':
                init.uniform_(model.weight.data, -init_weights, init_weights)
                init.uniform_(model.bias.data, -init_weights, init_weights)
            elif init_type == "conv_xavier_normal_":
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_normal_(m.weight)
                        nn.init.constant_(m.bias, init_weights)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_w)


# sample using triangle point picking
def sample_surface(faces, vertices, cnt):
    bsize, nvs, _ = vertices.shape
    
    weight, normal = get_weight_normal(faces, vertices)
    weight_sum = torch.sum(weight, dim = 1)
    dist = torch.distributions.categorical.Categorical(probs = weight / weight_sum[:, None])
    face_index = dist.sample((cnt,))

    tri_origins = vertices[:, faces[:, 0], :]
    tri_vectors = vertices[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((bsize, len(faces), 2, 3))

    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, cnt, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, cnt, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    random_lengths = torch.rand(cnt, 2, 1, device=vertices.device, dtype=tri_vectors.dtype)


    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    samples = sample_vector + tri_origins

    normals = torch.gather(normal, dim=1, index=face_index)

    return samples.float(), normals.float()

def get_weight_normal(faces, vertices):
    normal = torch.cross(vertices[:, faces[:, 1], :] - vertices[:, faces[:, 0], :],
                               vertices[:, faces[:, 2], :] - vertices[:, faces[:, 1], :], dim=2)
    weight = torch.norm(normal, dim=2)

    normal = normal / weight[:, :, None]

    weight = weight/2
    return weight, normal

def local_nonuniform_penalty(mesh):
    area = mesh_area(mesh)
    diff = area[mesh.gfmm][:, 0:1] - area[mesh.gfmm][:, 1:]
    penalty = torch.norm(diff, dim=1, p=1)
    loss = penalty.sum() / penalty.numel()
    return loss

def mesh_area(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    area = torch.cross(v1, v2, dim=-1).norm(dim=-1)
    return area

def manifold_upsample(mesh, save_path, manifold_path, num_faces=2000, res=3000, simplify=True):
    
    fname = save_path + 'recon_' + str(len(mesh.faces)) + '.obj'
    export(mesh, fname)

    temp_file = os.path.join(save_path, random_file_name('obj'))
    opts = ' ' + str(res) if res is not None else ''

    manifold_script_path = os.path.join(manifold_path, 'manifold')
    if not os.path.exists(manifold_script_path):
        raise FileNotFoundError(f'{manifold_script_path} not found')
    cmd = "{} {} {}".format(manifold_script_path, fname, temp_file + opts)
    os.system(cmd)

    if simplify:
        cmd = "{} -i {} -o {} -f {}".format(os.path.join(manifold_path, 'simplify'), temp_file,
                                                             temp_file, num_faces)
        os.system(cmd)

    m_out = Mesh(temp_file, hold_history=True, device=mesh.device)
    export(m_out, save_path + 'recon_' + str(len(m_out.faces)) + '_after.obj')
    [os.remove(_) for _ in list(glob.glob(os.path.splitext(temp_file)[0] + '*'))]
    return m_out


def random_file_name(ext, prefix='temp'):
    return f'{prefix}{uuid.uuid4()}.{ext}'


def copy_vertices(mesh):
    # print(mesh.vertices.shape[0])
    verts = torch.rand(1, mesh.vertices.shape[0], 3).to(mesh.vertices.device)
    # print(mesh.edges)
    x = verts[:, mesh.edges, :]
    # print(mesh.vertices.device)
    return x.view(1, mesh.ecnt, -1).permute(0, 2, 1).type(torch.float32)


class SimpleGrad(torch.autograd.Function):
    @staticmethod
    def forward(t, x):
        return x

    @staticmethod
    def backward(t, x):
        x[x != x] = 0
        return x


def array_times(num: int, iterable):
        return [i * num for i in iterable]

def make3(array):
    diff = [i % 3 for i in array]
    return [array[i] - diff[i] for i in range(len(array))]