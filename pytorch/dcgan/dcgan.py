import argparse
import os

import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import online_mean_and_std, plot_batch, device, normalized_dataloader_from_ImageFolder

# Set the random seed:
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# dataroot will be the path to the root of the dataset
dataroot = "/home/diego/Documents/datasets/mnist_png/training/"

# workers the number of worker threads for loading the data with DataLoader
workers = 2

# batch_size is the batch size during training (paper: 128)
batch_size = 128

# image_size is the image size:
image_size = 28

# nc is the number of color channels in the input channels (MNIST is grayscale)
nc = 1

# nz is the length of the latent vector
nz = 100

# ngf is the size of the feature maps in the generator (same as image size,
# as this will save us the trouble of changing things later on)
ngf = 28

# ndf is the size of the feature maps in the discriminator (ibidem)
ndf = 28

# num_epochs is the number of training epochs
num_epochs = 5

# lr is the learning rate for the optimizers (paper: 0.0002)
lr = 0.0002

# beta1 hyperparameter for the Adam optimizer (paper: 0.5)
beta1 = 0.5

# ngpu is the number of gpus available
ngpu = 0

#### Data ####

"""
MNIST data is located in:
path/to/mnist_png
    -> training
        -> 0
            -> 1.png
            -> 21.png
            ...
        -> 1
            ...
        ...
        -> 9
            ...
    -> testing
        -> 0
            -> 3.png
            -> 10.png
            ...
        -> 1
            ...
        ...
        -> 9
            ...
"""

# Create the dataset
dataset = dset.ImageFolder(
    root=dataroot,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=workers
)

dataloader = normalized_dataloader_from_ImageFolder(dataroot=dataroot,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=workers)

# Decide which device you wish to run on:
device = device(ngpu)

# Plot some of the training images:
plot_batch(dataloader=dataloader, size=8)


#### Implementation #####
