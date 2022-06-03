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
        # print(recall)
        precision = float(sum(d<threshold for d in dist1)) /float(len(dist1))
        # print(precision)
        fscore = 2*recall*precision / (recall+precision)
    # print(fscore)
    return fscore


def main():
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty1/recon_20950.obj',0.01))
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/poisson/giraffe_empty1_poisson.obj',0.01))
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty2/recon_20978.obj',0.01))
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/poisson/giraffe_empty2_poisson.obj',0.01))
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty3/recon_20976_after.obj',0.01))
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/poisson/giraffe_empty3_poisson.obj',0.01))
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/poisson/giraffe_empty0_poisson.obj',0.01))
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty0/recon_20896_after.obj',0.01))
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/data/author_giraffe_0.obj',0.01))
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty4/recon_10062_after.obj',0.01))
    print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/poisson/giraffe_empty4_poisson.obj',0.01))
if __name__ == '__main__':
    main()