#!/isr/bin/env python3
# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

print("""
# #####################################################################################################################
#  RNN - Creating Batches with Data
# #####################################################################################################################
""")
PI = 3.1416

x = torch.linspace(0, 799, 800)
y = torch.sin(x * 2 * PI / 40)

# Plot the sign wave.
plt.figure(figsize=(12, 4))
plt.xlim(-10, 801)
plt.grid(True)
plt.plot(y.numpy())
# plt.show()

# Train-test split for forecast model. Therefore, do not shuffle the dataset to retain the order or any pattern of the
# dataset.
test_size = 40
train_set = y[:-test_size]
test_set = y[-test_size:]

# Plot the training set.
plt.figure(figsize=(12, 4))
plt.xlim(-10, 801)
plt.grid(True)
plt.plot(train_set.numpy())
# plt.show()


# Split the train dataset again to feed in to the network as batches to mimic sequences of the dataset. Kind of a
# required thing in RNN.
def input_data(seq, ws):

    out = []  # [([0, 1, 2, 3], [4]), ([1, 2, 3, 4], [5])]  ... Tuple with "sequence" and "label respectively.

    L = len(seq)
    for i in range(L - ws):
        window = seq[i: i + ws]
        label = seq[i + ws]     # Get by the exact index hence we predict one point into the future.
        out.append((window, label))

    return out


window_size = 40
training_data = input_data(train_set, window_size)

print(len(training_data))
print(training_data[1])
print(training_data[1][0].view(40, 1, -1))

print("""
# #####################################################################################################################
#  Basic RNN - Creating LSTM (Long Short Term Memory) Model
# #####################################################################################################################
""")


class LSTM(nn.Module):
    """
    :param input_size - how many y value per timestamp we feed in. In this case it's 1, because the input training
    dataset has single column data sets, even though there are number of batches of them.

    :param hidden_size - how many neurons/perceptrons actually required in LSTM layer.

    :param output_size - how many predictions produce from the network. In this case it's 1 as we predict only one
    value into the future.
    """
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        '''
        LSTM layer takes input size (how many data points takes in) and how many neurons should create in the LSTM 
        layer. Here, the 'input_size' denotes the number of features in the dataset. 
        '''
        self.lstm = nn.LSTM(input_size, hidden_size)
        '''
        Linear layer to squeeze the output of LSTM layer neurons (50 in this matter) into less number of predicted
        values (only a single value in this case) base on the 'output_size'.
        '''
        self.linear = nn.Linear(hidden_size, output_size)
        '''
        Declare a initial state for hidden state and cell state that goes inside of all LSTM neurons for the first time 
        process. This is a 3-dimensional tensor, because we're passing two values (hidden state and cell state) for the 
        each LSTM cell of hidden layer.
        
        In a nutshell, this is to setup initial hidden state and cell state of all the neurons in LSTM layer and
        eventually we start to update these values as well as we go through the network.
        '''
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, sequence):

        lstm_out,  self.hidden = self.lstm(sequence.view(len(sequence), 1, -1), self.hidden)

        pred = self.linear(lstm_out.view(len(sequence), -1))

        return pred[-1]


# Setup a random seed to get same values with the lecture.
torch.manual_seed(42)

model = LSTM()
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# How many parameters we have here?
count = 0
for param in model.parameters():
    count += param.numel()
print(count)

print("""
# #####################################################################################################################
#  Basic RNN - Training and Forecasting
# #####################################################################################################################
""")


def train_test():
    epochs = 15
    # Number of points into the future.
    future = 40

    for i in range(epochs):

        for seq, y_train in training_data:
            # Reset the hyper-parameters.
            optimizer.zero_grad()
            # Reset the hidden and cell sates back to zeros.
            model.hidden = (torch.zeros(1, 1, model.hidden_size), torch.zeros(1, 1, model.hidden_size))

            y_pred = model.forward(seq)
            loss = criterion(y_pred, y_train.view(-1))

            loss.backward()
            optimizer.step()

        print(f'epoch: {i}, loss: {loss.item()}')

        # Grab the very-last window from the training_set(size 760). So the very first number we predict will be the
        # very last number of this list.
        preds = train_set[-window_size:].tolist()

        for i in range(future):
            seq = torch.FloatTensor(preds[-window_size:])

            with torch.no_grad():
                # Reset the hidden and cell sates back to zeros.
                model.hidden = (torch.zeros(1, 1, model.hidden_size), torch.zeros(1, 1, model.hidden_size))

                preds.append(model(seq).item())

            loss = criterion(torch.tensor(preds[-window_size:]), test_set)

        print(f'Performance of test range: {loss}')

        plt.figure(figsize=(12, 4))
        plt.xlim(700, 801)
        plt.grid(True)
        plt.plot(y.numpy())
        plt.plot(range(760, 800), preds[window_size:])
        plt.show()

# train_test()

# #####################################################################################################################
#  TRAIN ENTIRE DATASET, THEN ACTUALLY PREDICT THE VALUES OF NEXT WINDOW.
# #####################################################################################################################


epoch = 10
window_size = 40
future = 40

all_data = input_data(y, window_size)
print(len(all_data))

start = time.time()

for i in range(epoch):

    loss = 0
    for seq, y_train in all_data:

        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size), torch.zeros(1, 1, model.hidden_size))

        y_pred = model(seq)
        loss = criterion(y_pred, y_train.view(-1))

        loss.backward()
        optimizer.step()

    print(f'epoch: {i}, loss: {loss}')

duration = time.time() - start
print(f'Duration: {duration/60} minutes')


# FORECAST INTO UNKNOWN FUTURE.
preds = y[-window_size:].tolist()

for i in range(future):

    seq = torch.FloatTensor(preds[-window_size:])

    with torch.no_grad():

        model.hidden = (torch.zeros(1, 1, model.hidden_size), torch.zeros(1, 1, model.hidden_size))

        preds.append(model(seq))

plt.figure(figsize=(12, 4))
plt.xlim(0, 800 + future)
plt.grid(True)
plt.plot(y.numpy())
plt.plot(range(800, 800 + future), preds[window_size:])
plt.show()
