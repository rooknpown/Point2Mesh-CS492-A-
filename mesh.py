import os
import numpy as np

class Mesh():

    def __init__(self, file):

        if not os.path.exists(file):
            print("Mesh file does not exist at: " + str(file))
            return

    
        self.edges = None

        self.vertices, self.faces = load_obj(file)
        self.normalize()
        
        self.v_mask = np.ones(len(self.vertices))



    def load_obj(self, file):

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

    def normalize(self):
        maxCoord = []
        minCoord = []
        xyzScale = []
        self.translation = []
        for i in range(3):
            maxCoord.append(self.vs[:,i].max())
            minCoord.append(self.vs[:,i].min())
            xyzScale.append(maxCoord[i] - minCoord[i])
        scale = max(xyzScale)
        for i in range(3):
            self.translation.append((-maxCoord[i] - minCoord[i])/2/scale)


        self.vertices /= self.scale
        self.vertices += self.translation[None, :]

        


class PartMesh():
    def __init__(self):