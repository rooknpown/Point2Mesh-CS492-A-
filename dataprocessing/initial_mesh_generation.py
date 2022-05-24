import argparse
import open3d

def out_path_type(out_path):
    if not out_path.endswith('.obj'):
        raise argparse.ArgumentTypeError('Out path should be in type .obj file')
    return out_path

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Generate Initial Mesh", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--pcd_path", required = True, type= str, dest="pcd_path")
  parser.add_argument("--out_path", required = True, type= out_path_type, dest="out_path")

  args = parser.parse_args()

  pcd_path = args.pcd_path
  out_path = args.out_path

  pointCld = open3d.io.read_point_cloud(pcd_path)
  hull, _ = pointCld.compute_convex_hull()
  open3d.io.write_triangle_mesh(out_path, hull)