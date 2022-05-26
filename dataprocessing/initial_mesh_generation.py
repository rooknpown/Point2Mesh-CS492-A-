import argparse
import open3d
import os

def out_path_type(out_path):
    if not out_path.endswith('.obj'):
        raise argparse.ArgumentTypeError('Out path should be in type .obj file')
    return out_path

def inplace_manifold(path, res: int, manifold_software_path):
    cmd = f'{manifold_software_path}/manifold {path} {path} {res}'
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Initial Mesh", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pcd_path", required = True, type= str, dest="pcd_path")
    parser.add_argument("--out_path", required = True, type= out_path_type, dest="out_path")
    parser.add_argument("--genus", required = False, type= int, dest="genus")
    args = parser.parse_args()

    pcd_path = args.pcd_path
    out_path = args.out_path
    genus = args.genus

    if genus is None or genus > 0:
        isgenus0 = False
    else:
        isgenus0 = True

    pointCld = open3d.io.read_point_cloud(pcd_path)
    if isgenus0 :
        hull, _ = pointCld.compute_convex_hull()
        open3d.io.write_triangle_mesh(out_path, hull)
    else:
        alpha = 0.5
        mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pointCld, alpha)
        open3d.io.write_triangle_mesh(out_path, mesh)
        inplace_manifold(out_path, 500, '/root/code/Manifold/build')

