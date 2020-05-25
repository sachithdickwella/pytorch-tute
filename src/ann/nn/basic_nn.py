#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F     # Activations functions coming from here.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

print("""
# #################################################################################################
#  Basic PyTorch Neural Network - Part One (1)
# #################################################################################################
""")


class Model(nn.Module):

    """
    :param in_features=4 is not arbitrary. It denotes the number of input features available.
    In this case 4 types of features.

    Input features: sepal length,sepal width,petal length and petal width.

    :param hidden_layer1=8 is an arbitrary number which pretty much the developer can play with.
    There is no right and wrong number of neurons/perceptrons.

    :param hidden_layer2=10 is an arbitrary number which pretty much the developer can play with.
    There is no right or wrong number of neurons/perceptrons.

    *** All of the hidden layers' neuron count can be defined arbitrary as developer chose, this would
    increase the intuition and the processing power required.

    :param out_features=3 is not arbitrary. It denotes the number of output classes required.
    In this case 3 types of classes.

    Output classes: Setosa, Versicolor and Virginica.
    """
    def __init__(self, in_features=4, hidden_layer1=8, hidden_layer2=10, out_features=3):
        super().__init__()
        """
        How many layers? (Can play around - More layers -> more processing requirement).
        Input layer (4 features) -> hidden layer1 (N neurons [can play around])
            -> hidden layer2 (N neuron [can play around]) -> output layer (3 classes)
        
        *** 'fc' variable name stands for 'Fully Connected' and developer can play around adding 
        and removing how many layers have in between input and output layers and how many neurons
        there are in between those hidden layers.
        """
        self.fc1 = nn.Linear(in_features, hidden_layer1)  # 'fc' variable name stands for 'Fully Connected'.
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))    # RELU - Rectified Linear Unit (Activation Function)
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


torch.manual_seed(32)
model = Model()

print(model)

print("""
# #################################################################################################
#  Basic PyTorch Neural Network - Part Two (2)
# #################################################################################################
""")

df = pd.read_csv('../../notebooks/Data/iris.csv')
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

print(type(X))
print(type(y))  # These are Panda Series. Therefore take values of them.

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()   # Because it's multi-class classification problem.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 'Adam' is efficient optimizer. In fact it's optimize learning rate as go on (Adaptive learning rate)
# See Theory lectures.
print(model.parameters)

# EPOCHS
epochs = 100
losses = []

for i in range(epochs):

    # Forward the 'X_train' data and get predicted 'y'.
    y_pred = model.forward(X_train)

    # Calculate loss/cost of predicted 'y' against the actual value 'y_train'.
    # No need to do "ONE HOT ENCODING" between 'y_pred' and 'y_train' hence we used 'CrossEntropyLoss'.
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if i % 10 == 0:
        print(f"epoch {i} and loss: {loss}")

    # Below three (3) statements perform back-propagation in the epoch.
    optimizer.zero_grad()   # Reset gradient to zero(0)
    loss.backward()
    optimizer.step()


plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('CrossEntropyLoss')
# plt.show()

print("""
# #################################################################################################
#  Basic PyTorch Neural Network - Part Three (3)
# #################################################################################################
""")

# Tensor just ignore gradient tracking in this block and after will continue the same as before.
with torch.no_grad():   # Closable operation. Disable 'gradient' engine.

    y_eval = model.forward(X_test)

    loss = criterion(y_eval, y_test)

print(loss)
# If the loss value is lot less than the training data sets' last loss values. Then the training
# data set is over-fitting.
correct = 0
# Check how many items have correctly guessed by the model.
with torch.no_grad():

    for i, data in enumerate(X_test):

        y_eval = model.forward(data)
        """
        Print a value 'tensor([-1.7047,  4.9455, -0.3859])' format and the highest value shows the likelihood
        which the network thinks that the class it belongs to. Three output, thus only has three(3) classes and 
        three(3) output neurons.
        """
        print(f'{i + 1}.) {y_eval} {y_test[i]}')

        if y_eval.argmax().item() == y_test[i]:
            correct += 1


print("\nCorrect results are %.2f" % (correct / len(y_test) * 100))

# Save the trained model in a file.
torch.save(model.state_dict(), "nn/my_iris_model.pt")  # Save the state-dictionary here.

# Open the saved model to use.
new_model = Model()  # This 'Model' has no idea what the 'weight' and 'bias' would be.
# Therefore load the saved model from the disk.
new_model.load_state_dict(torch.load("nn/my_iris_model.pt"))
print(new_model.eval())

correct = 0
with torch.no_grad():

    for i, data in enumerate(X_test):

        y_eval = new_model.forward(data)
        print(f"{i + 1}.) {y_eval} {y_test[i]}")

        if y_eval.argmax().item() == y_test[i]:
            correct += 1

# Correct results form saved model.
print("\nCorrect results are {:.2f}%".format(correct / len(y_test) * 100))
# *** LOADING and SAVING model dictionary, assumes the availability of original model class. ***
# If need to save the entire model instead of trained weights and biases should use the 'model'
# instance instead of 'model.state_dict()'
torch.save(model, 'nn/model_file.pt')
# and load with torch.load(str) function.
model_saved = torch.load('nn/model_file.pt')
print(model_saved.eval())

# Applying unknown data to the network and get results.
mystery_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])

labels = ['Iris setosa', 'Iris virginica', 'Iris versicolor', 'Mystery iris']

with torch.no_grad():
    res = new_model(mystery_iris)
    print(res)
    print(labels[res.argmax()])
