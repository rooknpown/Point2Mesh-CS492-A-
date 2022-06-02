import torch 
import numpy as np

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

def load_obj(file):

    vertices = []
    faces = []
    f = open(file)
    for line in f:
        line = line.strip().split()

        if not line:
            continue
        # example line: v 0.716758 0.344326 -0.568914

        elif line[0] == 'v':
            coordList = []
            for x in line[1:4]:
                coordList.append(float(x))
            vertices.append(coordList[:])

        # example line: f 1 4 2
        elif line[0] == 'f':
            coordList = []
            for x in line[1:4]:
                coordList.append(int(x) - 1)
            faces.append(coordList[:])
    f.close()
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices, faces

def normalize(vertices):
    maxCoord = []
    minCoord = []
    xyzScale = []
    translation = []
    for i in range(3):
        maxCoord.append(vertices[:,i].max())
        minCoord.append(vertices[:,i].min())
        xyzScale.append(maxCoord[i] - minCoord[i])
    scale = max(xyzScale)
    for i in range(3):
        translation.append((-maxCoord[i] - minCoord[i])/2/scale)


    vertices /= scale
    vertices += [translation]
    return translation, scale, vertices
