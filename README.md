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
Thingi10k\cite{https://doi.org/10.48550/arxiv.1605.04797}: {\small \url{https://ten-thousand-models.appspot.com/}}
COSEG\cite{wang2012active}: {\small\url{http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm}}
A Large Dataset of Object Scans\cite{choi2016large}{\small\url{http://redwood-data.org/3dscan/}}
Point2Mesh \cite{Hanocka_2020} {\small\url{https://github.com/ranahanocka/point2mesh/blob/master/scripts/get_data.sh}}


Borrowed code for
watertight manifold \cite{https://doi.org/10.48550/arxiv.1802.01698} :{\small\url{https://github.com/hjwdzh/Manifold}}
mesh and MeshCNN \cite{Hanocka_2020}, \cite{hanocka2019meshcnn}: {\small\url{https://github.com/ranahanocka/point2mesh}} {\small\url{https://github.com/ranahanocka/MeshCNN}}
Specifically, **The codes in authors/ folder are codes modified from original code https://github.com/ranahanocka/Point2Mesh/ and https://github.com/ranahanocka/MeshCNN**.


Everything else are implemented ourselves.
