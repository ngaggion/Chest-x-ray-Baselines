source activate torch

python trainer.py --name Hybrid --model HybridGNet
python trainer.py --name FC --model FC
python trainer.py --name PCA_lungs --model PCA --lungs