import hydra
from mesh import Mesh

@hydra.main(config_path=".", config_name="run.yaml")
def main(config):
    mesh = Mesh()
    


if __name__ == '__main__':
    main()