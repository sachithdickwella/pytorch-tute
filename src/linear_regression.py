#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

print('''
# #################################################################################################
# Linear Regression with PyTorch - Part One
# #################################################################################################
''')

X = torch.linspace(1, 50, 50).reshape(-1, 1)
print(X)

torch.manual_seed(71)   # To get same random numbers.

e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
print(e)

y = 2 * X + 1 + e
print(y.shape)

# Convert tensor to numpy.
n = X.numpy()

# Plot the chart.
plt.scatter(X.numpy(), y.numpy())  # OR plt.scatter(X.numpy())

# KEEP THAT IN MIND, PRECEDING SIMPLE LINEAR REGRESSION DOESN'T KEEP/TRACK
# GRADIENT/DERIVATIVE SINCE WE HAVEN'T ENABLE IT.

# Let's see how this linear simple model without gradient pre-selects weight and bias in random.
torch.manual_seed(59)       # Keep exact same noise profile (same random numbers).

model = nn.Linear(in_features=1, out_features=1)    # Simple linear model with single input and output.

print(model.weight)     # Automatically setup the 'required_grad=True'.
print(model.bias)


# Setup model class.
class Model(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_predicted = self.linear(x)
        return y_predicted


print('''
# #################################################################################################
# Linear Regression with PyTorch - Part Two
# #################################################################################################
''')


class Model2(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_predicted = self.linear(x)
        return y_predicted


# Using the 'Model' class.
torch.manual_seed(59)

model = Model(in_features=1, out_features=1)
print(model.linear.weight)
print(model.linear.bias)

# Iterate through all the model parameters at once.
for name, param in model.named_parameters():
    print("name: {n}\tparam: {p}".format(n=name, p=param.item()))

# Call the model by passing a tensor.
x = torch.tensor([2.])
print(model.forward(x))

# Check how to perform this simple Linear model against random integers between 0 - 50 predicting y.
x1 = torch.linspace(0., 50., 50)
print(x1)
# First manual 'y' calculation with custom weigh and bias.
w1 = 0.10597813129425049    # Very first weigh declaration.
b1 = 0.9637961387634277     # Very first bias declaration.

y1 = w1 * x1 + b1    # Calculating output.
print(y1)

plt.scatter(X.numpy(), y.numpy())
plt.plot(x1, y1, 'r')   # Last argument is color of the graph.
# This predictions actually doesn't learn anything and use only the same weight and bias throughout the tensor.
# There should have loss/cost function calculate the error between input and output (Mean Square Error - MSE) to
# update weight and bias to learn. In order to apply these update should use this simple NN model above created.

# USING LINEAR NEURAL NETWORK - Model class.
# Writing loss function or using existing PyTorch loss/cost function to get Mean Squared Error (MSE).
criterion = nn.MSELoss()        # Basic Loss/Cost function calculate how fare off we are from the actual value.
# This is the Mean Squared Error (MSE) loss function. By Convention, MSE label as 'criterion'. Because 'criteria'
# or 'criterion' EVALUATE the network performance.
print(criterion)

# OPTIMIZATION - STOCHASTIC GRADIENT DESCENT with applied learning rate. Learning rate tells the optimizer, how much
# to adjust each parameter on next round of calculations.
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)   # Stochastic Gradient Descent - Learning Rate 0.001 is a
# good learning rate to begin with. For more intricate neural network, might take some time to play with learning rate
# to adjust. Should do experimentation with learning to realize optimum learning rate value.

# EPOCH = Single pass through the entire dataset. In order to train the model, should choose number of epochs. Should
# choose sufficiently large number of epochs to reach a 'PLATEAU'(minimum).
epochs = 50  # This value is quite large for this kind of simple problem.
losses = []  # To keep track of mean squared error as go along.

# Training the model.
for i in range(epochs):
    i += 1

    # PREDICTING ON THE FORWARD PASS.
    y_pred = model.forward(X)

    # CALCULATING LOSS (ERROR).
    loss = criterion(y_pred, y)

    # RECORD THAT ERROR.
    losses.append(loss)

    print(f"epoch: {i}, loss: {loss.item()}, weight: {model.linear.weight.item()}, bias: {model.linear.bias.item()}")

    # Gradient accumulate on each new epoch round. So reset the gradient on each round of epoch.
    optimizer.zero_grad()

    # Common steps to training a model every time.
    loss.backward()     # Back propagate on the loss (Final output).
    optimizer.step()    # Update the hyper-parameters of the model. In this case single weight and single bias.


# RECORDED losses can be used to plot the learning rate graph against the EPOCHS.
plt.plot(range(epochs), losses)
plt.xlabel('EPOCHS')
plt.ylabel('MSE Loss')

# Test the trained model against some data.
x = np.linspace(0., 50., 50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()

y_prediction = current_weight * x + current_bias

# Plot predicted data.
plt.plot(x, y_prediction)
