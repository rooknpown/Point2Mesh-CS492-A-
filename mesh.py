import os

class Mesh():

    def __init__(self, file):

        if not os.path.exists(file):
            print("Mesh file does not exist at: " + str(file))
            return

        self.vertices = None
        self.edges = None
        

        


class PartMesh():
    def __init__(self):