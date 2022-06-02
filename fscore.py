import open3d as o3d
import argparse
import os

def fscore(s,t,threshold):
    s=o3d.io.read_point_cloud(s)
    mesh_t=o3d.io.read_triangle_mesh(t)
    t = mesh_t.sample_points_uniformly(number_of_points=25000)

    dist1 = s.compute_point_cloud_distance(t)
    dist2 = t.compute_point_cloud_distance(s)

    if len(dist1) and len(dist2):
        recall = float(sum(d<threshold for d in dist2)) /float(len(dist2))
        print(recall)
        precision = float(sum(d<threshold for d in dist1)) /float(len(dist1))
        print(precision)
        fscore = 2*recall*precision / (recall+precision)
    print(fscore)
    return fscore


def main():
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty0/recon_23246_after.obj',0.03))

if __name__ == '__main__':
    main()