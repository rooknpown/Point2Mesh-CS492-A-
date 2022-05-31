import hydra
import numpy as np
import torch
from authors.mesh import Mesh, SubMesh
from model import PriorNet
from torch import optim
# from loss import BeamGapLoss
from myloss import bidir_chamfer_loss, BeamGapLoss
import time
import os
import uuid
import glob
import os
from authors.utils import get_weight_normal, sample_surface, local_nonuniform_penalty, mesh_area, manifold_upsample, copy_vertices
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@hydra.main(config_path=".", config_name="run.yaml")
def main(config):
    pcpath = config.get("pcpath")
    initmesh = config.get("initmeshpath")
    savepath = config.get("savepath")
    bfs_depth = config.get("bfs_depth")
    iters = config.get("iters")
    beamgap_iter = config.get("beamgap-iter")
    beamgap_mod = config.get("beamgap-mod")
    norm_weight = config.get("norm_weight")
    max_face = config.get("max_face")
    manifold_res = config.get("manifold_res")
    manifold_path = config.get("manifoldpath")
    disable_net = config.get("disable_net")

    torch.manual_seed(5)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    mesh = Mesh(initmesh, hold_history=True, device = device)
    # print(mesh.vertices)
    coords, normals = read_pcs(pcpath)
    
    # normalize coordinates and normals
    print(mesh.scale)
    print(mesh.translation)
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
    # print(len(sub_mesh.init_vertices))
    net = init_net(mesh, sub_mesh, device,
                    in_channel = config.get("in_channel"),
                    convs = config.get("convs"), 
                    pool= config.get("pools"), 
                    res_blocks = config.get("res_blocks"), 
                    leaky = config.get("leaky"), 
                    transfer = config.get("trasnfer"),
                    init_weights = config.get("init_weights"),
                    disable_net = disable_net)
    optimizer = optim.Adam(net.parameters(), lr = config.get("learning_rate"))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x : 1 - min((0.1*x / float(iters), 0.95)))
    rand_verts = copy_vertices(mesh)
    # print("rand_verts")
    # print(rand_verts)

    # beamgap_loss = BeamGapLoss(config.get("thres"), device)
    # beamgap_loss.update_params(sub_mesh, torch.cat([coords, normals], dim=-1))

    samples = config.get("samples")
    start_samples = config.get("start_samples")
    upsample = config.get("upsample")
    slope = config.get("slope")
    diff = (samples - start_samples) / int(slope * upsample)

    export_period = config.get("export_period")
    for i in range(iters):
        num_sample = int(diff * min(i%upsample, slope*upsample)) + start_samples
        start_time = time.time()
        # print("BBB")
        K = net(rand_verts, sub_mesh)
        # print(K)
        for sub_i, vertices in enumerate(K):
            # print("rand_verts s: ")
            # print(rand_verts.shape)
            # print("iter: " + str(i) + "subi: " + str(sub_i))
            optimizer.zero_grad()
            # print(vertices[0])
            sub_mesh.update_vertices(vertices[0], sub_i)
            num_sample = int(diff * min(i%upsample, slope*upsample)) + start_samples
            new_xyz, new_normals = sample_surface(sub_mesh.base_mesh.faces, 
                                                sub_mesh.base_mesh.vertices.unsqueeze(0),
                                                num_sample)
            # print("CCCCCCCCCCCCCCCCCCCCCC")
            # print(new_xyz)
            # print(coords)
            
            
            if (i <beamgap_iter) and (i % beamgap_mod):
                loss = beamgap_loss(sub_mesh, sub_i)
            else:
                # xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(new_xyz, coords, 
                #                                                      x_normals = new_normals, y_normals = normals, unoriented = True) 
                xyz_chamfer_loss, normals_chamfer_loss = bidir_chamfer_loss(new_xyz, coords, new_normals, normals)
                loss = xyz_chamfer_loss + norm_weight * normals_chamfer_loss
            
            # bidir_chamfer_loss(new_xyz, coords, new_normals, normals)
            # print("rand_verts m: ")
            # print(rand_verts.shape)
            
            loss += 0.1 * local_nonuniform_penalty(sub_mesh.base_mesh).float() 

            loss.backward()
            optimizer.step()
            scheduler.step()
            sub_mesh.base_mesh.vertices.detach_()
            # print("rand_verts e: ")
            # print(rand_verts.shape)
            # print("DDDD")
        end_time = time.time()
        # print("NNN")
        print("iter: " + str(i) + "/" + str(iters) + " loss: " + str(loss.item()) +
            " num samples: " + str(num_sample) + " time:" + str(end_time - start_time))
        # print("MM")
        # sub_mesh.base_mesh.print()
        if i % export_period ==0 and i != 0:
            with torch.no_grad():
                sub_mesh.build_base_mesh()
                export(sub_mesh.base_mesh, savepath + "recon_iter_" + str(i) + ".obj")


        if i != 0 and i + 1 % upsample == 0:
            num_faces = int(np.clip(len(sub_mesh.base_mesh.faces)*1.5, len(sub_mesh.base_mesh.faces), max_face))

            if num_faces > len(mesh.faces) or True:
                mesh = manifold_upsample(mesh, savepath, manifold_path, num_faces = min(num_faces, max_face),
                                        res = manifold_res)
                mesh.print()
                # print("AAAA: " + str(mesh.ecnt))
                sub_mesh = SubMesh(mesh, get_num_submesh(len(mesh.faces)), bfs_depth = bfs_depth)
                print("upsampled to " + str(len(mesh.faces)) + "num parts: " + str(sub_mesh.num_sub))
                net = init_net(mesh, sub_mesh, device,
                    in_channel = config.get("in_channel"),
                    convs = config.get("convs"), 
                    pool= config.get("pools"), 
                    res_blocks = config.get("res_blocks"), 
                    leaky = config.get("leaky"), 
                    transfer = config.get("trasnfer"),
                    init_weights = config.get("init_weights"),
                    disable_net = disable_net)
                optimizer = optim.Adam(net.parameters(), lr = config.get("learning_rate"))
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x : 1 - min((0.1*x / float(iters), 0.95)))
                rand_verts = copy_vertices(mesh)

                if i < beamgap_iter:
                    beamgap_loss.update_params(sub_mesh, coords)
    
    with torch.no_grad():
        export(mesh, savepath +'last_recon.obj')

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
            res_blocks, leaky, transfer, init_weights, disable_net):
    init_vertices = mesh.vertices.clone().detach()
    net = PriorNet(sub_mesh = sub_mesh, in_channel = in_channel, convs = convs, pool = pool, 
                    res_blocks = res_blocks, leaky = leaky, transfer = transfer, 
                    init_weights = init_weights, init_vertices = init_vertices, disable_net = disable_net).to(device)
    return net

def export(mesh, path):
    vertices = mesh.vertices.cpu().clone()
    vertices -=  torch.tensor([mesh.translation])
    vertices *= mesh.scale
    print("exporting!!!")
    with open(path, 'w+') as fil:
        for vi, v in enumerate(vertices):
            fil.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in mesh.faces:
            fil.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))

if __name__ == '__main__':
    main()