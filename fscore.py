import open3d as o3d
import argparse
import os
import numpy as np

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
        if recall + precision == 0:
            return 0
        fscore = 2*recall*precision / (recall+precision)
    # print(fscore)
    return fscore

def calc_poisson (name, thres):
    result = []
    input_pct =  '/root/p2m/data/' + name[:-6] + '.ply'
    for i in range(5):
        poisson_path =  '/root/p2m/poisson/' + name + str(i) + '_poisson.obj'
        result.append(fscore(input_pct, poisson_path, thres))
    result = np.array(result)
    print("poisson on " + name)
    print(result)
    print("fscore avg: " + str(np.mean(result)))
    print("fscore std: " + str(np.std(result)))


def calc_p2m (name, thres):
    facedict = {'giraffe_empty':[20896, 20950, 20978, 20976, 20980]
                , 'bull_empty':[17026, 16668, 18314, 17246, 20938]
                ,'guitar_noise':[23716, 21654, 23420, 23124, 23460]
                , "tiki_noise": [55000, 51432, 51308, 52000, 51308]}

    result = []
    input_pct =  '/root/p2m/data/' + name[:-6] + '.ply'
    for i in range(5):
        p2m_path = '/root/p2m/checkpoints/' + name + str(i) + '/recon_' + str(facedict[name][i]) + '_after.obj'
        # print(p2m_path)
        result.append(fscore(input_pct, p2m_path, thres))
    result = np.array(result)
    print("p2m on " + name)
    print(result)
    print("fscore avg: " + str(np.mean(result)))
    print("fscore std: " + str(np.std(result)))
    
def main():
    thres = 0.007
    calc_poisson('giraffe_empty', thres)
    calc_p2m('giraffe_empty', thres)
    calc_poisson('guitar_noise', thres)
    calc_p2m('guitar_noise', thres)
    thres = 0.1
    calc_poisson('tiki_noise', thres)
    calc_p2m('tiki_noise', thres)
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/poisson/giraffe_empty0_poisson.obj',0.01))
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty0/recon_20896_after.obj',0.01))
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty1/recon_20950_after.obj',0.01))
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/poisson/giraffe_empty1_poisson.obj',0.01))
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty2/recon_20978_after.obj',0.01))
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/poisson/giraffe_empty2_poisson.obj',0.01))
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty3/recon_20976_after.obj',0.01))
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/poisson/giraffe_empty3_poisson.obj',0.01))
    
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/data/author_giraffe_0.obj',0.005))
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/checkpoints/giraffe_empty4/recon_20980_after.obj',0.005))
    # print(fscore('/root/p2m/data/giraffe.ply','/root/p2m/poisson/giraffe_empty4_poisson.obj',0.005))


if __name__ == '__main__':
    main()