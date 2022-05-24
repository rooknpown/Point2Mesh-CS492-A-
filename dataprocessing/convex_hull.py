import trimesh
import argparse
import pathlib
import warnings
import hydra
import os

@hydra.main(config_path=".", config_name="convex_hull.yaml")
def run(config):
    

    if config.get("genus0"):
        xyz, _ = read_pcs(config.get("input_path"))
        m = trimesh.convex.convex_hull(xyz)
        vs, faces = m.vertices, m.faces
    
    else:
        vs, faces = load_obj(config.get("input_path"))
        

    outpath = config.get("output_path")
    export(outpath, vs, faces)
    isblender = config.get("blender")
    if isblender:
        blender_rehull(outpath, outpath, config.get("blender_res"), config.get("blender_path"))
    else:
        inplace_manifold(outpath, config.get("manifold_res"), config.get("manifold_path"))

    num_faces = count_faces(outpath)
    if isblender:
        num_faces /= 2
    num_faces = int(num_faces)
    if num_faces < config.get("faces"):
        software = 'blender' if isblender else 'manifold'
        warnings.warn(f'only {num_faces} faces where generated by {software}. '
                      f'try increasing --{software}-res to achieve the desired target of {config.get("faces")} faces')
    else:
        inplace_simplify(outpath, config.get("faces"), config.get("manifold_path"))

    print('*** Done! ****')


def export(path, vertices, faces):
    with open(path, 'w+') as fil:
        for vi, v in enumerate(vertices):
            fil.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in faces:
            fil.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))


def count_faces(path) -> int:
    with open(path, 'r') as file:
        lines = file.read().split('\n')
    return sum(map(lambda x: x.startswith('f'), lines))


def inplace_manifold(path, res: int, manifold_software_path):
    cmd = f'{manifold_software_path}/manifold {path} {path} {res}'
    os.system(cmd)


def inplace_simplify(path, faces: int, manifold_software_path):
    cmd = f'{manifold_software_path}/simplify -i {path} -o {path} -f {faces}'
    os.system(cmd)


def blender_rehull(target, dest, res: int, blender_path):
    base_path = pathlib.Path(__file__).parent.absolute()
    cmd = f'{blender_path}/blender --background --python {base_path}/blender_scripts/blender_hull.py' \
          f' {target} {res} {dest} > /dev/null 2>&1'
    os.system(cmd)

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


if __name__ == '__main__':
    run()