import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

def create_train_dataloader(): ->

    pass

def create_val_dataloader():
    pass

if __name__ == '__main__':

    data_paths = ['../../../data/example.npz']

    dataset_maker = TrainValDataloaderMaker(data_paths)

    dl_train, dl_val = dataset_maker.make_dataloader()

    print(dl_val)

