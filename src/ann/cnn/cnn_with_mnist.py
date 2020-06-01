#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

x = X_train[0:1]     # Get the first element from the batch with slicing to retain the shape/dimension.
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
# Shape becomes ([1, 16, 11, 11]) hence  loosing the each border information by 1px, due to
# no padding and the kernel size. In the MNIST dataset, there are no information around the
# borders.
print(x.shape)

x = pool1(x)    # Run through the same max-pool.
# Shape becomes ([1, 16, 5, 5]) hence the max-pool the 3x3 kernel by 2 strides.
print(x.shape)

# Finally feed output to a Fully Connected (fc) layer (Flatten the output of pool).
x = x.view(1, -1)
print(x.shape)

# Fully connected layer to forward feed the pool output.
fc1 = nn.Linear(x.shape[1], 10)
print(fc1)

# Feed to FC layer.
x = fc1(x)
print(x)

# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################

print("""
# #####################################################################################################################
#  MNIST with CNN Code Along - Part Two (2)
# #####################################################################################################################
""")


class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        pass

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = self.pool1(X)
        X = F.relu(self.conv2(X))
        X = self.pool1(X)
        X = F.relu(self.fc1(X.view(-1, 5 * 5 * 16)))
        X = F.relu(self.fc2(X))
        X = F.log_softmax(self.fc3(X), dim=1)

        return X


# Set the seed to match with lecturer.
torch.manual_seed(42)

# Create the model instance.
model = ConvolutionalNetwork()
print(model)

# Check whether parameters have been decreased.
all = 0
for param in model.parameters():
    all += param.numel()
    print(param.numel())

print(f"Sum: {all}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training of the model.
start = time.time()
# TRACKING VARIABLES
epochs = 5
training_losses = []
train_correct = []
test_losses = []
test_correct = []
# LOOP EPOCHS
for i in range(epochs):

    trn_crt = 0
    tst_crt = 0

    # TRAIN
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        y_pred = model.forward(X_train)     # NO FLATTEN
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred, dim=1)[1]
        batch_crt = (predicted == y_train).sum()    # True(1) / False(0) sum
        trn_crt += batch_crt

        # back propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 600 == 0:
            print(f'epoch: {i}, batch: {b} -> loss {loss.item()}')

    train_correct.append(trn_crt)
    test_losses.append(loss)

    # TEST
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            y_pred = model.forward(X_test)

            predicted = torch.max(y_pred, dim=1)[1]
            batch_crt = (predicted == y_test).sum()
            tst_crt += batch_crt

    loss = criterion(y_pred, y_test)

    test_correct.append(tst_crt)
    test_losses.append(loss)


duration = time.time() - start
print(f'Training took {duration / 60} minutes')

print("""
# #####################################################################################################################
#  MNIST with CNN Code Along - Part Three (3) [Model Evaluation]
# #####################################################################################################################
""")

# Check the file 'cnn_with_mnist.ipynb' to see rest. with plotting ↑.
