import hydra
import numpy as np
import torch
from mesh import Mesh, SubMesh
from model import PriorNet
from torch import optim
from loss import BeamGapLoss
import time
from pytorch3d.loss import chamfer_distance

@hydra.main(config_path=".", config_name="run.yaml")
def main(config):
    pcpath = config.get("pcpath")
    initmesh = config.get("initmeshpath")
    savepath = config.get("savepath")
    bfs_depth = config.get("bfs_depth")
    iters = config.get("iters")

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    mesh = Mesh(initmesh, device)
    coords, normals = read_pcs(pcpath)
    
    # normalize coordinates and normals
    coords = normalize(coords, mesh.scale, mesh.translation)
    normals = np.array(normals, dtype=np.float32)

    # give to gpu
    coords = torch.Tensor([coords]).type(torch.float32).to(device)
    normals = torch.Tensor([normals]).type(torch.float32).to(device)

    num_submesh = get_num_submesh(len(mesh.faces))
    # print("111111111111111111111111111111111")
    # print(coords.shape)
    # print(normals.shape)
    # print(num_submesh)
    sub_mesh = SubMesh(mesh, num_submesh, bfs_depth = bfs_depth)
    # print(sub_mesh.num_sub )
    print(len(sub_mesh.init_vertices))
    net = init_net(mesh, sub_mesh, device,
                    in_channel = config.get("in_channel"),
                    convs = config.get("convs"), 
                    pool= config.get("pools"), 
                    res_blocks = config.get("res_blocks"), 
                    leaky = config.get("leaky"), 
                    transfer = config.get("trasnfer"),
                    init_weights = config.get("init_weights"))
    optimizer = optim.Adam(net.parameters(), lr = config.get("learning_rate"))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x : 1 - min((0.1*x / float(iters), 0.95)))
    rand_verts = copy_vertices(mesh).to(device)
    print("rand_verts")
    print(rand_verts)

    loss = BeamGapLoss(config.get("thres"), device)

    samples = config.get("samples")
    start_samples = config.get("start_samples")
    upsample = config.get("upsample")
    slope = config.get("slope")
    diff = (samples - start_samples) / int(slope * upsample)

    for i in range(iters):
        num_sample = int(diff * min(i%upsample, slope*upsample)) + start_samples
        start_time = time.time()
        optimizer.zero_grad()
        for sub_i, vertices in enumerate(net(rand_verts, sub_mesh)):
            sub_mesh.update_vertices(vertices, sub_i)
            new_xyz, new_normals = sample_surface(sub_mesh.base_mesh.faces, 
                                                sub_mesh.base_mesh.vertices.unsqueeze(0),
                                                num_sample)
            print("CCCCCCCCCCCCCCCCCCCCCC")
            print(new_xyz)
            print(coords)
            xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(new_xyz, coords, 
                                                                     x_normals = new_normals, y_normals = normals)
            


def read_pcs(file):

    coords = []
    normals = []
    f = open(file)
    for line in f:
        line = line.strip().split()
        if(len(line) == 6):
            coord = [line[0],line[1],line[2]]
            coords.append(coord[:])
            normal = [line[3],line[4],line[5]]
            normals.append(normal[:])

    return coords, normals

def normalize(coords, scale, translation):
    coords = np.array(coords, dtype=np.float32)
    coords /= scale
    coords += [translation]

    return coords

def get_num_submesh(num_faces):
    subnumList = [1, 2, 4, 8]
    faceBin = [8000, 16000, 20000]
    num_submesh = subnumList[np.digitize(num_faces, faceBin)]
    return num_submesh

def init_net(mesh, sub_mesh, device, in_channel, convs, pool, 
            res_blocks, leaky, transfer, init_weights):
    init_vertices = mesh.vertices.clone().detach()
    net = PriorNet(sub_mesh = sub_mesh, in_channel = in_channel, convs = convs, pool = pool, 
                    res_blocks = res_blocks, leaky = leaky, transfer = transfer, 
                    init_weights = init_weights, init_vertices = init_vertices).to(device)
    return net

def copy_vertices(mesh):
    verts = torch.rand(1, mesh.vertices.shape[0], 3).to(mesh.vertices.device)
    x = verts[:, mesh.edges, :]
    return x.view(1, mesh.ecnt, -1).permute(0, 2, 1).type(torch.float64)

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

if __name__ == '__main__':
    main()