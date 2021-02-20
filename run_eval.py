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
import sys
import importlib
import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from dataloader.DataLoaders import MatterportDepthDataLoader
from modules.losses import *

# Fix CUDNN error for non-contiguous inputs
import torch.backends.cudnn as cudnn

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cudnn.enabled = True
cudnn.benchmark = True

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-mode', action='store', dest='mode', default='eval', help='"test","eval" or "train" mode')
parser.add_argument('-exp', action='store', dest='exp', default='exp_msg_matter', help='Experiment name as in workspace directory')
parser.add_argument('-chkpt', action='store', dest='chkpt', default=36, nargs='?', #default=None
                    help='Checkpoint number to load')
parser.add_argument('-set', action='store', dest='set', default='val', type=str, nargs='?',
                   help='Which set to evaluate on "val", "selval" or "test"')
args = parser.parse_args()

# Path to the workspace directory
training_ws_path = './workspace/'
exp = args.exp
exp_dir = training_ws_path + exp + '/'

# Add the experiment's folder to python path
sys.path.append(exp_dir)

# Read parameters file
with open(os.path.join(exp_dir, 'params.json'), 'r') as fp:
    params = json.load(fp)
params['gpu_id'] = 0

# Use GPU or not
device = torch.device("cuda:" + str(params['gpu_id']) if torch.cuda.is_available() else "cpu")
# Dataloader for Matterport
dataloaders, dataset_sizes = MatterportDepthDataLoader(params)
# Import the network file
f = importlib.import_module('network_' + exp)
model = f.network().to(device)
model = nn.DataParallel(model)

# Import the trainer
t = importlib.import_module('trainers.' + params['trainer'])

mode = args.mode # train    eval
# sets = ['val'] #  train  selval'train',
if mode == 'eval':
    sets = [args.set] #  train  selval
if mode == 'test':
    sets = ['test']

with torch.cuda.device(params['gpu_id']):
    # Objective function
    objective = locals()[params['loss']]()

    # Optimize only parameters that requires_grad
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # The optimizer
    optimizer = getattr(optim, params['optimizer'])(parameters, lr=params['lr'],
                                                    weight_decay=params['weight_decay'])

    # Decay LR by a factor of 0.1 every exp_dir7 epochs
    lr_decay = lr_scheduler.StepLR(optimizer, step_size=params['lr_decay_step'], gamma=params['lr_decay'])
    mytrainer = t.KittiDepthTrainer(model, params, optimizer, objective, lr_decay, dataloaders, dataset_sizes,
                                    workspace_dir=exp_dir,  sets=sets, use_load_checkpoint=args.chkpt)

    if __name__ == '__main__':
        if mode == 'train':
            # train the network
            net = mytrainer.train(params['num_epochs'])  #
        elif mode == 'eval':
            net = mytrainer.evaluate()
        else:
            net = mytrainer.test_fn()

