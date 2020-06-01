#!/usr/bin/env python3
# -*- coding: utf8 -*-

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

print("""
# #####################################################################################################################
#  CIFAR-10 Dataset with CNN - Code Along - Part One (1)
# #####################################################################################################################
""")

# Download and setup datasets.
transform = transforms.ToTensor()
train_data = datasets.CIFAR10(root='../../../../../../notebooks/Data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='../../../../../../notebooks/Data', train=False, download=True, transform=transform)

print(train_data)
print(test_data)

torch.manual_seed(101)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# Define strings for labels.
class_names = ['plane', '  car', ' bird', '  cat', ' deer', '  dog', ' frog', 'horse', ' ship', 'truck']

# Grab first batch of images.
for images, labels in train_loader:
    break

print(labels)

# Print the labels
print('Label:', labels.numpy())
print('Class: ', *np.array([class_names[i] for i in labels]))

# Draw a grid of first batch.
im = make_grid(images, nrow=5)
plt.figure(figsize=(10, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

print("""
# #####################################################################################################################
#  CIFAR-10 Dataset with CNN - Code Along - Part Two (2)
# #####################################################################################################################
""")


class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=6 * 6 * 16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = self.pool1(X)
        X = F.relu(self.conv2(X))
        X = self.pool1(X)
        X = F.relu(self.fc1(X.view(-1, 6 * 6 * 16)))
        X = F.relu(self.fc2(X))
        X = F.log_softmax(self.fc3(X), dim=1)
        return X


torch.manual_seed(101)
model = ConvolutionalNetwork()
print(model)

# Check the parameters.
s = 0
for param in model.parameters():
    s += param.numel()

print(f"Number of parameters: {s}")

# Train the model and rest of the details continue on 'cnn_with_cifar10.ipynb' file.
