#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

print("""
# #################################################################################################
#  ANN with MNIST - Part One(1) - Using Linear ANN.
# #################################################################################################
""")

# MNIST IMAGE --> Tensor
transform = transforms.ToTensor()

# Load dataset.
train_data = datasets.MNIST(root='../../../../../../notebooks/Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../../../../../../notebooks/Data', train=False, download=True, transform=transform)

print(train_data)
print(test_data)

# Check the type and length.
print(type(train_data))     # torchvision.datasets.mnist.MNIST
print(len(train_data))

# Details of a single element.
print(type(train_data[0]))  # Tuple
print(train_data[0])        # Tuple -> (Tensor, Label(int))

# Observe internal of the tuple.
image, label = train_data[0]
# ([1, 28, 28]) -> 1 - grayscale image, 28 x 28 is the size of the image by pixels.
# First value of the shape would 0 or 1 hence this is grayscale, which represents
# black or white. Sometimes could be -1 to 1 instead of 0 to 1.
print(image.shape)
print(label)    # label value is just integer scalar value.

# Plot the image - This shows the colored images with default color mapping called 'viridis'.
# To see this in actual grayscale, provide a custom color mapping. In order to find 'cmap' values,
# refer the matplot documentation.

# plt.imshow(image.reshape(28, 28), cmap='gist_yarg')   # Drop the extra dimension.
# plt.show()

# To take randomly shuffled batch of data from original data set.
torch.manual_seed(101)
# Shuffle dataset and load them using data loader to avoid picking same set of images.
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

# #################################################################################################
# Visualize what derived by 'DataLoader' above - NOT MUCH IMPORTANT
# #################################################################################################
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))  # FORMATIING

# FIRST BATCH.
for images, labels in train_loader:
    break
# Preceding statement just grab the first batch and break the loop. Cannot assign directly to get the first
# batch hence *loader is iterable.
print(images.shape)
# 4D Tensor -> ([100, 1, 28, 28]) - Here it's 100 images (mind the batch size of loader), of 1 color (grayscale),
# 28 x 28 images
print(labels.shape)  # ([100]) - Just 100 corresponding labels.

# Print first 12 labels and images.
print('Labels: ', labels[:12].numpy())  # Formats , hence the 'np.set_printoptions()' above.

im = make_grid(images[:12], nrow=12)
print(im.shape)
#plt.figure(figsize=(10, 4))

# Transpose the image across the diagonal CWH to HWC (Color Channel, Width, Height -> Height, Width, Color Channel).
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

# #################################################################################################
# #################################################################################################
# #################################################################################################

print("""
# #################################################################################################
#  ANN with MNIST - Part Two(2) - Using Linear ANN (784 Perceptrons [IN]).
# #################################################################################################
""")


class MultilayerPerceptron(nn.Module):

    def __init__(self, in_sz=784, out_sz=10, layers=(120, 84)):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)  # MULTI CLASS CLASSIFICATION


# Set the seed.
torch.manual_seed(101)
# Initialize the  the model.
model = MultilayerPerceptron()
print(model)
# ###########################################################################################################
# ###########################################################################################################
# Note:
#   Later when we discuss about Convolutional Neural Networks(CNN) against Artificial Neural Networks
#   (ANN), we'll see the mail motivation behind the idea of CNN. CNNs are just;
#
#       * Takes less parameters in to perform training.
#       * Much more efficient than ANN on images data processing.
#       * Lot less parameters.
#
#   Above multilayer perceptron model has 784 inputs and the next layer has 120 neurons are pretty much
#   larger than it takes in CNN.
#
# ###########################################################################################################
# ###########################################################################################################
for param in model.parameters():
    print(param.numel())

print(f'{sum([param.numel() for param in model.parameters()])}')  # 105,214 total parameters.
# Lot less parameters in CNN.

# Loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Flatten out the images [100, 1, 28, 28] -> [100, 784].
print(images.shape)

viewed = images.view(100, -1)    # Retain first dimension and reorient the rest automatically thus -1.
print(viewed.shape)

print("""
# #################################################################################################
#  ANN with MNIST - Part Two(3) - Using Linear ANN (Training).
# #################################################################################################
""")

epochs = 10

# TRACKERS
train_losses = []
test_loss = []
train_correct = []
test_correct = []


def train():
    start_time = time.time()

    for i in range(epochs):

        trn_correct = 0
        tst_correct = 0
        loss = 0

        for b, (X_train, y_train) in enumerate(train_loader):

            b += 1

            y_pred = model.forward(X_train.view(100, -1))
            loss = criterion(y_pred, y_train)

            # torch.max(x, dim=1) would return a max object. max_object = (values=Tensor, indices=Tensor), also access by
            # indexes like a list.
            predicted = torch.max(y_pred.data, dim=1)[1]  # Get the indices tensor from torch.max object.
            batch_correct = (predicted == y_train).sum()  # Filter match status to boolean tensor and sum 'True' values.
            trn_correct += batch_correct  # Append correct matches to 'training_correct' variable.

            # Back propagate.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 200 == 0:
                accuracy = trn_correct.item() * 100 / (100 * b) # Multiply by 100 hence the batch size is 100.
                print(f'epoch: {i} batch: {b} -> loss: {loss.item()}, accuracy: {accuracy}%')

        train_losses.append(loss)
        train_correct.append(trn_correct)

        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                y_val = model.forward(X_test.view(500, -1))

                predicted = torch.max(y_val.data, dim=1)[1]
                tst_correct += (predicted == y_test).sum()

        loss = criterion(y_val, y_test)
        test_loss.append(loss)
        test_correct.append(tst_correct)

    print(f'Duration: {(time.time() - start_time) / 60:0.2f} minutes')


train()
# torch.save(model.state_dict(), 'mnist_trained_model.pt')
# model = model.load_state_dict(torch.load('mnist_trained_model.pt'))

print("""
# #################################################################################################
#  ANN with MNIST - Part Two(3) - Using Linear ANN (Evaluation).
# #################################################################################################
""")

# Refer the Jupyter Notebook files in this folder. ↑
