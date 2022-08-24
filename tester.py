from models.HybridGNet2IGSC import Hybrid 

import os 
import numpy as np
from torchvision import transforms
import torch

from utils.utils import scipy_to_torch_sparse, genMatrixesLungs, genMatrixesLungsHeart
import scipy.sparse as sp

import cv2
import pathlib
import re

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]



def getDenseMask(RL, LL, imagesize = 1024):
    img = np.zeros([1024,1024])
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 255, -1)
    img = cv2.drawContours(img, [LL], -1, 255, -1)

    return img

if __name__ == "__main__":
    data_root = pathlib.Path("../Dataset_Cande/Images")
    all_files = list(data_root.glob('*.jpg'))
    all_files = [str(path) for path in all_files]
    all_files.sort(key = natural_key)

    print(len(all_files))

    device = "cuda:0"

    A, AD, D, U = genMatrixesLungs()
    N1 = A.shape[0]
    N2 = AD.shape[0]

    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()

    D_ = [D.copy()]
    U_ = [U.copy()]

    config = {}

    config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to('cuda:0') for x in X] for X in (A_, D_, U_))

    config['latents'] = 64
    config['inputsize'] = 1024

    f = 32
    config['filters'] = [2, f, f, f, f//2, f//2, f//2]
    config['skip_features'] = f

    hybrid = Hybrid(config.copy(), D_t, U_t, A_t).to(device)
    hybrid.load_state_dict(torch.load("Training/Hybrid_lungs/bestMSE.pt"))
    hybrid.eval()
    print('Model loaded')

    contador = 0
    with torch.no_grad():
        for image in all_files:    
            print('\r',contador+1,'of', len(all_files),end='')
            img = cv2.imread(image, 0) / 255.0
            data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device).float()

            output = hybrid(data)
            if len(output) > 1:
                output = output[0]
            output = output.cpu().numpy().reshape(-1, 2) 

            RL = output[:44] * 1024
            LL = output[44:94] * 1024
            outseg = getDenseMask(RL, LL)
            
            cv2.imwrite(image.replace("Images","Masks"),outseg)
            contador+=1