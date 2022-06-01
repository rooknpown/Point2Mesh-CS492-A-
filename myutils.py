import torch 


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