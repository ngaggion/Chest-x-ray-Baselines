# Baselines for Chest x-ray Landmark Segmentation Dataset.

Baselines are taken from [our latest work](https://arxiv.org/abs/2203.10977) and trained under both lungs and heart segmentation and only lungs segmentation.\
By default we refer as HybridGNet the best performing variation presented on the paper, with 2 Image-To-Graph Skip Connection (IGSC) layers.

Results are available on Jupyter notebooks, train and test subsets are provided on .txt files and training scripts are available.

## Citation

N. Gaggion, L. Mansilla, C. Mosquera, D. H. Milone and E. Ferrante, "Improving anatomical plausibility in medical image segmentation via hybrid graph neural networks: applications to chest x-ray analysis," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3224660.

https://doi.org/10.1109%2Ftmi.2022.3224660

```
@article{Gaggion_2022,
	doi = {10.1109/tmi.2022.3224660},
	url = {https://doi.org/10.1109%2Ftmi.2022.3224660},
	year = 2022,
	publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
	author = {Nicolas Gaggion and Lucas Mansilla and Candelaria Mosquera and Diego H. Milone and Enzo Ferrante},
	title = {Improving anatomical plausibility in medical image segmentation via hybrid graph neural networks: applications to chest x-ray analysis},
	journal = {{IEEE} Transactions on Medical Imaging}
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