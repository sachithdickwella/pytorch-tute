#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

print("""
# #####################################################################################################################
# RNN on a Time Series - Part One(1)
# #####################################################################################################################
""")

# With index_col=0, make the first column the index column and with parse_dates=True, try to parse the index column
# as datetime values.
df = pd.read_csv('../../../notebooks/Data/TimeSeriesData/Alcohol_Sales.csv', index_col=0, parse_dates=True)
print(df)
print(df.columns)
print(len(df))

# Drop null values if any.
df = df.dropna()
print(len(df))

# Plot out current dataset.
df.plot(figsize=(12, 4))

# Since the numbers in the dataset are 'object' type grab and convert the required column to 'float' type.
y = df.iloc[:, 0].values.astype(float)
print(y, end='\n\n')

# Define train and test sizes (1 year has 12 data-points).
test_size = 12
train_set = y[:-test_size]
test_set = y[-test_size:]

# Normalize the data sets in order for faster and accurate convergence and void data leakage between train
# and test data sets..
scaler = MinMaxScaler(feature_range=(-1, 1))
# Normalization has two steps.
#   1. fitting - Grab the minimum and max values from the data sets.
#   2. transform - perform the transformation on normalization.
scaler.fit(train_set.reshape(-1, 1))
# Preceding statement mark the min and max values for data sets.
train_norm = scaler.transform(train_set.reshape(-1, 1))
# Perform the actual transformation on dataset.
# To perform preceding two-steps in a single statement.
train_norm = scaler.fit_transform(train_set.reshape(-1, 1))
print(train_norm[:10])
print(len(train_norm))

# Convert the numpy array to tensor.
train_norm = torch.tensor(train_norm).view(-1)
train_norm = train_norm.float()
print(train_norm)

window_size = 12    # Per year.


def input_data(seq, ws):

    out = []
    L = len(seq)

    for i in range(L - ws):
        window = seq[i:i + ws]
        label = seq[i + ws:i + ws + 1]
        out.append((window, label))
    return out


train_data = input_data(train_norm, window_size)
print(len(train_data))
print(train_data[0])

print("""
# #####################################################################################################################
# RNN on a Time Series - Part Two(2)
# #####################################################################################################################
""")


class LSTMNetwork(nn.Module):

    def __init__(self, in_features=1, hidden_size=100, out_features=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=3)
        self.linear = nn.Linear(in_features=hidden_size, out_features=out_features)
        self.hidden = (torch.zeros(3, 1, hidden_size), torch.zeros(3, 1, hidden_size))

    def forward(self, seq):
        lstm, _ = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        seq = self.linear(lstm.view(len(lstm), -1))
        return seq[-1]


torch.manual_seed(101)
model = LSTMNetwork()
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model.
epoch = 50

start = time.time()
for i in range(epoch):
    i += 1
    loss = 0
    for seq, label in train_data:
        model.hidden = (torch.zeros(3, 1, model.hidden_size), torch.zeros(3, 1, model.hidden_size))
        optimizer.zero_grad()

        y_pred = model(seq)
        loss = criterion(y_pred, label)

        loss.backward()
        optimizer.step()

    print(f'epoch: {i}, loss: {loss}')

duration = time.time() - start
print(f'Duration: {duration / 60} minutes')

# De-normalize the training values and test with test dataset.
future = 12

preds = train_norm[-window_size:].tolist()
model.eval()
for f in range(future):
    f += 1

    seq = torch.tensor(preds[-window_size:])
    with torch.no_grad():

        model.hidden = (torch.zeros(3, 1, model.hidden_size), torch.zeros(3, 1, model.hidden_size))
        y_pred = model(seq)

        preds.append(y_pred)

# Inverse the normalization.
true_prediction = scaler.inverse_transform(np.array(preds[-window_size:])).reshape(-1, 1)

# These true predictions and actual values should be even almost.
print(true_prediction, y[-window_size:].view(-1, 1))

# Create range for date and time with numpy.
x = np.arange('2018-12-01', '2019-02-02', dtype='datetime64[M]')
print(x)

# Plot the predictions.
fig = plt.figure(figsize=(12, 4))
plt.grid(True)
plt.autoscale(axis='x', tight=True)
fig.autofmt_xdate()
plt.plot(df.iloc[:, 0])
plt.plot(x, true_prediction)
plt.show()

# Go for IPython notebook file hence the time it takes for single execution. ↑
