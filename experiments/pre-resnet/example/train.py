from config import config, args
from dataset import create_train_dataloader, create_val_dataloader
from network import create_network

from utils.misc import save_args, save_checkpoint, load_checkpoint
from training.train import train_one_epoch, eval_one_epoch

import torch
import json
import time
import numpy as np
from tensorboardX import SummaryWriter
import argparse

import os
from collections import OrderedDict

DEVICE = torch.device('cpu')
torch.backends.cudnn.benchmark = True

net = create_network(config.num_channels)
net.to(DEVICE)
criterion = config.create_loss_function().to(DEVICE)

optimizer = config.create_optimizer(net.parameters())
lr_scheduler = config.create_lr_scheduler(optimizer)


dataset_maker = TrainValDataloaderMaker(config.data_paths)

print('--- Dataset Loaded. Totoal: {} samples'.format(dataset_maker.num))
ds_train = create_train_dataloader()
ds_val = create_val_dataloader()


now_epoch = 0

if args.auto_continue:
    args.resume = os.path.join(config.model_dir, 'last.checkpoint')
if args.resume is not None and os.path.isfile(args.resume):
    now_epoch = load_checkpoint(args.resume, net, optimizer,lr_scheduler)

while True:
    if now_epoch > config.num_epochs:
        break
    now_epoch = now_epoch + 1

    descrip_str = 'Training epoch:{}/{} -- lr:{}'.format(now_epoch, config.num_epochs,
                                                                       lr_scheduler.get_lr()[0])
    train_one_epoch(net, ds_train, optimizer, criterion, DEVICE,
                    descrip_str, )
    if config.val_interval > 0 and now_epoch % config.val_interval == 0:
        eval_one_epoch(net, ds_val, DEVICE, )

    lr_scheduler.step()

    save_checkpoint(now_epoch, net, optimizer, lr_scheduler,
                    file_name = os.path.join(config.model_dir, 'epoch-{}.checkpoint'.format(now_epoch)))
