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
conda activate ptom
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
Borrowed data from

Thingi10k https://ten-thousand-models.appspot.com/

COSEG http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm

A Large Dataset of Object Scans http://redwood-data.org/3dscan/

Point2Mesh https://github.com/ranahanocka/point2mesh/blob/master/scripts/get_data.sh


Borrowed code for

watertight manifold https://github.com/hjwdzh/Manifold

mesh :https://github.com/ranahanocka/point2mesh

and MeshCNN : https://github.com/ranahanocka/MeshCNN

Specifically, **The codes in authors/ folder are codes modified from original code https://github.com/ranahanocka/Point2Mesh/ and https://github.com/ranahanocka/MeshCNN**.


Everything else are implemented ourselves.


4.4. Irreproducible experiments
Some experiments of the original paper [4] were not reproducible due to lack of information or resource constraint.
Experiments of Figure 3, 16, 17 of the original paper did not have enough explanation on the algorithm to reproduce theresults. 
For Figure 11 we didn’t have a scanner nor the datafrom it. 
The table at page 11 was not reproducible as the specific 50 configurations were not provided.
Also, classifying noisy or detailed was done manually by the author which is qualitative and likely to be different from our standard.
