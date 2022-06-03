import open3d as o3d
import numpy as np

in_f = f'guitar_noise0'
file=f'/root/p2m/data/{in_f}.ply'
out = f'/root/p2m/poisson/{in_f}_poisson.obj'
pcd = o3d.io.read_point_cloud(file)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
o3d.io.write_triangle_mesh(out, mesh)