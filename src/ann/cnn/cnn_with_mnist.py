#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("""
# #####################################################################################################################
#  MNIST with CNN Code Along - Part One (1)
# #####################################################################################################################
""")

# Load MNIST Data.
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='../../../../../../notebooks/Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../../../../../../notebooks/Data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# #####################################################################################################################
# Convolutional NN breakdown.
# #####################################################################################################################
# 1 - Color channel, 6 - Output channels(FILTERS), 3 - 3x3 Kernel(FILTER SIZE), 1 - stride by 1.
conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)
# 6 - Input filters(Kernels), 16 - Filters(Kernels), 3 - 3x3 Kernel size, 1 - stride by 1.
conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
# 2 - Kernel size, 2 - stride by 2.
pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # OR just use F.max_pool2d(tensor, kernel_size, stride).

# Extract single image and feedforward manually without activation.
for i, (X_train, y_train) in enumerate(train_loader):
    break

print(X_train.shape)

x = X_train[0:1]     # Get the first element from the batch with slicing to retain the shape.
print(x.shape)
print(x)

# Passing through Convolutional units and activation functions.
x = F.relu(conv1(x))
# Shape becomes ([1, 6, 26, 26]) hence loosing the each border information by 1px, due to no
# padding and kernel size. In the MNIST dataset, there are no information around the borders.
print(x.shape)

x = pool1(x)
# Shape becomes ([1, 6, 13, 13]) hence max-pool the 3x3 kernel by 2 strides.
print(x.shape)

x = F.relu(conv2(x))
# Shape becomes ([1, 16, 11, 11]) hence the loosing the each border information by 1px, due to
# no padding and the kernel size. In the MNIST dataset, there are no information around the borders.
print(x.shape)

x = pool1(x)    # Run through the same max-pool.
# Shape becomes ([1, 16, 5, 5]) hence the max-pool the 3x3 kernel by 2 strides.
print(x.shape)

# Finally feed output to a Fully Connected (fc) layer (Flatten the output of pool).
x = x.view(1, -1)
print(x)

# Fully connected layer to forward feed the pool output.
fc1 = nn.Linear(x.shape[1], 10)
print(fc1)

# Feed to FC layer.
x = fc1(x)
print(x)

# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################