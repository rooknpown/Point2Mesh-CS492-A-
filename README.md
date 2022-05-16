# Point2Mesh-CS492-A-
Point2Mesh implementation for KAIST 2022 Spring CS492(A)

First clone the repo

```
git clone https://github.com/rooknpown/Point2Mesh-CS492-A-.git
cd Point2Mesh-CS492-A-
```
Set environment running below

```
conda env create -f environment.yml
conda activate p2m
```

Install Manifold here
https://github.com/hjwdzh/Manifold


Set paths and parameters in run.yaml
pcpath: path for point cloud ply file
initmeshpath: path for initial mesh of the given object
savepath: path for saving mesh output 
manifoldpath: path for the Manifold installed above. set the build folder location (~/Manifold/build)

Run the code by

```
python run.py

```
