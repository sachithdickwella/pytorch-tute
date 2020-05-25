#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Model class to perform the training.
class Model(nn.Module):

    def __init__(self, in_features=4, hidden_layer1=8, hidden_layer2=10, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


model = Model()
print(model)

# Read data from CSV and structure them.
df = pd.read_csv("../../../notebooks/Data/iris.csv")
print(df.head())

X_values = df.drop('target', axis=1).values
y_values = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.2, random_state=40)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Train the model.
torch.manual_seed(71)
criterion = nn.CrossEntropyLoss()  # Hence multi-class classification.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # Adam optimizer.

epochs = 110
losses = []

for i in range(epochs):

    y_predictions = model.forward(X_train)

    loss = criterion(y_predictions, y_train)
    losses.append(loss)

    if i % 10 == 0:
        print(f"epoch {i} -> loss {loss}")

    # back-propagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Plot the training rate to see the convergence.
plt.plot(range(epochs), losses)
plt.xlabel("EPOCH")
plt.ylabel("Cross Entropy Loss")
# plt.show()

# Test the trained model with test data.
with torch.no_grad():

    y_pred = model.forward(X_test)

    loss = criterion(y_pred, y_test)

print(loss, end='\n\n')
# Loss value cannot deviate much from the last training loss. If it is, training data set is over-fitting.

# Just evaluate the test data without gradient tracking.
with torch.no_grad():

    correct = 0
    for i, data in enumerate(X_test):

        y_pred = model.forward(data)

        print(f"{i + 1}.) {y_pred} -> {y_test[i]}")

        # See how many are correct
        if y_pred.argmax().item() == y_test[i]:
            correct += 1

print("Correct results are {:.1f}%".format(correct / len(y_test) * 100))

# Get the classification for given unknown entry.
labels = ['Iris setosa', 'Iris virginica', 'Iris versicolor', 'Mystery iris']
unknown_iris = torch.tensor([[5.6, 3.7, 2.2, 0.5]])

with torch.no_grad():
    result = model(unknown_iris)
    print(result)

    print(labels[result.argmax().item()])
