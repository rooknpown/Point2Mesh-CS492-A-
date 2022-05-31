import torch
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_objs_as_meshes, save_ply
import os
from os import listdir
from os.path import isfile, join
import hydra

## sample points from mesh
@hydra.main(config_path=".", config_name="mesh_sampler.yaml")
def sample_points_all(config):
    
    num_samples = config.get("num_samples")
    input_path = config.get("input_path")
    output_path = config.get("output_path")
    print(num_samples)

    input_objs = [input_path + f for f in listdir(input_path) if isfile(join(input_path, f))]
    output_plys = [output_path + f[:-4] + '.ply' for f in listdir(input_path) if isfile(join(input_path, f))]
    print(input_objs)
    print(output_plys)
    for i in range(len(input_objs)):
        sample_points(input_objs[i], output_plys[i], num_samples)


def sample_points(input_obj, output_ply, num_samples):
    
    input_obj_list = [input_obj]

    mesh = load_objs_as_meshes(input_obj_list)
    coords, normals = sample_points_from_meshes(mesh, num_samples = num_samples, return_normals = True)
    save_ply(output_ply, verts = coords[0, :], verts_normals = normals[0, :], ascii = True)
    print("saved to:" + output_ply)
    


if __name__ == '__main__':
    sample_points_all()