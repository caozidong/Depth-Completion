import torch
import os
import cv2
import numpy as np
from utils.visualize_sparse import colorize,colorize_roi
from matplotlib import pyplot as plt
import torch.nn.functional as F

def saveTensorToImage(t, idxs, save_to_path, rgb):
    if os.path.exists(save_to_path) == False:
        os.mkdir(save_to_path)

    for i in range(t.size(0)):
        if not rgb:
            im = t[i, 0, :, :].detach().data.cpu().numpy()
            # np.save(os.path.join(save_to_path, str(idxs[i].data.cpu().numpy()).zfill(10) + '.npy'), im)
            colorize(im,  0, os.path.join(save_to_path, str(idxs[i].data.cpu().numpy()).zfill(10) + '.png'), 0)
        else:
            im = t[i, :, :, :].data.cpu().numpy() * 255
            im = np.transpose(im, (1, 2, 0)).astype(int)
            plt.imsave(os.path.join(save_to_path, str(idxs[i].data.cpu().numpy()).zfill(10) + '.png'), im)