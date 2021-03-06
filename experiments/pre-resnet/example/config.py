from easydict import EasyDict
import sys
import os
import argparse
import numpy as np
import torch

def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-3])
lib_dir = os.path.join(root_path, 'lib')
add_path(lib_dir)

from training.config import TrainingConfigBase, SGDOptimizerMaker, \
    PieceWiseConstantLrSchedulerMaker

class TrainingConfing(TrainingConfigBase):

    lib_dir = lib_dir

    num_epochs = 10
    val_interval = 2

    create_optimizer = SGDOptimizerMaker(lr =1e-1, momentum = 0.9, weight_decay = 1e-3)
    create_lr_scheduler = PieceWiseConstantLrSchedulerMaker(milestones = [5, 7, 9], gamma = 0.1)

    create_loss_function = torch.nn.CrossEntropyLoss


config = TrainingConfing()

config.data_paths = ['../../../data/example.npz']
config.num_channels = 20

# About data
# C.inp_chn = 1
# C.num_class = 10

parser = argparse.ArgumentParser()

parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                 help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                 metavar='N', help='mini-batch size')
parser.add_argument('-d', type=int, default=0, help='Which gpu to use')
parser.add_argument('-adv_coef', default=1.0, type = float,
                    help = 'Specify the weight for adversarial loss')
parser.add_argument('--auto-continue', default=False, action = 'store_true',
                    help = 'Continue from the latest checkpoint')
args = parser.parse_args()


if __name__ == '__main__':
    pass
