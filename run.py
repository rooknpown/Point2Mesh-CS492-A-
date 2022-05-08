import hydra
import numpy as np
import torch
from mesh import Mesh, SubMesh
from model import PriorNet

@hydra.main(config_path=".", config_name="run.yaml")
def main(config):
    pcpath = config.get("pcpath")
    initmesh = config.get("initmeshpath")
    savepath = config.get("savepath")
    bfs_depth = config.get("bfs_depth")

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
    sub_mesh = SubMesh(mesh, num_submesh, bfs_depth = bfs_depth)
    init_net(mesh, sub_mesh, device,
            in_channel = config.get("in_channel"),
            convs = config.get("convs"), 
            pool= config.get("pools"), 
            res_blocks = config.get("res_blocks"), 
            leaky = config.get("leaky"), 
            transfer = config.get("trasnfer"),
            init_weights = config.get("init_weights"))



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
                    init_weights = init_weights, init_vertices = init_vertices) 

if __name__ == '__main__':
    main()