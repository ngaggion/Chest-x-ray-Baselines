# Baselines for Chest x-ray Landmark Segmentation Dataset.

Baselines are taken from our latest work and trained under both lungs and heart segmentation and only lungs segmentation.\
Results are available on Jupyter notebooks, train and test subsets are provided on .txt files and training scripts are available.

## Citation

Gaggion, N., Mansilla, L., Mosquera, C., Milone, D. H., & Ferrante, E. (2022). Improving anatomical plausibility in medical image segmentation via hybrid graph neural networks: applications to chest x-ray analysis. arXiv preprint arXiv:2203.10977.

https://arxiv.org/abs/2203.10977

```
@misc{https://doi.org/10.48550/arxiv.2203.10977,
  doi = {10.48550/ARXIV.2203.10977},
  url = {https://arxiv.org/abs/2203.10977},
  author = {Gaggion, Nicolás and Mansilla, Lucas and Mosquera, Candelaria and Milone, Diego H. and Ferrante, Enzo},
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Improving anatomical plausibility in medical image segmentation via hybrid graph neural networks: applications to chest x-ray analysis},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Installation:

First create the anaconda environment:

```
conda env create -f environment.yml
```
Activate it with:
```
conda activate torch
```

In case the installation fails, you can build your own enviroment.

Conda dependencies: \
-PyTorch 1.10.0 \
-Torchvision \
-PyTorch Geometric \
-Scipy \
-Numpy \
-Pandas  \
-Scikit-learn \
-Scikit-image 

Pip dependencies: \
-medpy==0.4.0 \
-opencv-python==4.5.4.60 