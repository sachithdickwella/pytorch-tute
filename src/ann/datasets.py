#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

print('''
# #################################################################################################
#  DataSets with PyTorch - Part One
# #################################################################################################
''')

df = pd.read_csv('../../notebooks/Data/iris.csv')
print(df.head())
print(df.shape)

# Plot the graphs with data set.
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()

plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris Setosa','Iris Virginica','Iris Versicolor']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
# plt.show()

# METHOD 1
# Data analysis using SciKit learn. Later on PyTorch DataSets. (SciKit is the traditional way!)
feature = df.drop('target', axis=1).values
label = df['target'].values

# Here 'test_size=0.2' is 20% of original dataset consider as test data set.
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train).reshape(-1, 1)
y_test = torch.LongTensor(y_test).reshape(-1, 1)

# print(y_train)

# METHOD 2
# Using PyTorch inbuilt utility classes DataSet and DataLoader and more intuitive.

data = df.drop('target', axis=1).values
label = df['target'].values

iris = TensorDataset(torch.FloatTensor(data), torch.LongTensor(label))
print(type(iris))
print(len(iris))

for train, test in iris:  # Iterate throughout the 'TensorDataSet'.
    print(train, test)


# To get these data shuffled or load by small batches wrap the 'TensorDataset' with 'DataLoader'.
iris_loader = DataLoader(iris, batch_size=50, shuffle=True)

for i_batch, sample_batch in enumerate(iris_loader):
    print(i_batch, sample_batch)
