source activate torch

python trainer.py --name PCA --model PCA 
python trainer.py --name Hybrid_lungs --model HybridGNet --lungs
python trainer.py --name FC_lungs --model FC --lungs
