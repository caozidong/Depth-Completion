"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dataloader.MatterportDataset import MatterportDataset
import random
import glob
import json
import numpy as np

def MatterportDepthDataLoader(params):
    norm_factor = params['data_normalize_factor']
    invert_depth = params['invert_depth']
    ds_dir = params['dataset_dir']
    rgb2gray = params['rgb2gray'] if 'rgb2gray' in params else False
    flip = params['flip'] if ('flip' in params) else False
    dataset = params['dataset'] if 'dataset' in params else 'KittiDepthDataset'
    num_worker = 8

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    ###### Training Set ######
    train_data_path = os.path.join(ds_dir)
    train_gt_path = os.path.join(ds_dir)
    train_rgb_path = os.path.join(ds_dir)

    # train_transform = transforms.Compose([transforms.CenterCrop((352, 1216))])

    image_datasets['train'] = eval(dataset)(train_data_path, train_gt_path, train_rgb_path, setname='train',
                                                transform=None, norm_factor=norm_factor,
                                                invert_depth=invert_depth,
                                                rgb2gray=rgb2gray, flip=flip)

    # Select the desired number of images from the training set
    if params['train_on'] != 'full':
        image_datasets['train'].data = image_datasets['train'].data[0:params['train_on']]  # file directions
        image_datasets['train'].gt = image_datasets['train'].gt[0:params['train_on']]

    dataloaders['train'] = DataLoader(image_datasets['train'], shuffle=True, batch_size=params['train_batch_sz'],
                                      num_workers=num_worker)
    dataset_sizes['train'] = {len(image_datasets['train'])}

    ###### Validation Set ######
    val_data_path = os.path.join(ds_dir)
    val_gt_path = os.path.join(ds_dir)
    val_rgb_path = os.path.join(ds_dir)

    # val_transform = transforms.Compose([transforms.CenterCrop((352, 1216))])

    image_datasets['val'] = eval(dataset)(val_data_path, val_gt_path, val_rgb_path, setname='val', transform=None,
                                              norm_factor=norm_factor, invert_depth=invert_depth,
                                              rgb2gray=rgb2gray, flip=flip)
    dataloaders['val'] = DataLoader(image_datasets['val'], shuffle=False, batch_size=params['val_batch_sz'],
                                    num_workers=num_worker)
    dataset_sizes['val'] = {len(image_datasets['val'])}

    print(dataset_sizes)

    return dataloaders, dataset_sizes
