#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

print("""
# #################################################################################################
#  Full ANN Code Along - Regression Part One(1) (Feature Engineering)
# #################################################################################################
""")

df = pd.read_csv('../../../notebooks/Data/NYCTaxiFares.csv')
print(df.head())

print(df['fare_amount'].describe())


def haversine_distance(df, src_lat, src_long, tar_lat, tar_long):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers

    phi1 = np.radians(df[src_lat])
    phi2 = np.radians(df[tar_lat])

    delta_phi = phi2 - phi1
    delta_lambda = np.radians(df[tar_long] - df[src_long])

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = (r * c)  # in kilometers
    return d


# FEATURE ENGINEERING - Taking a existing features already have, create new more useful features
# than original features.

# Engineering distance between two points with co-ordinates.
df['dist_km'] = haversine_distance(df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
                                   'dropoff_longitude')
print(df['dist_km'].head())

print(df.info())

# Engineering datetime object(string) to be DateTime object.
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
print(df.info())
print(df.head())

my_time = df['pickup_datetime'][0]
print(my_time)
print(my_time.hour)

df['EDTDate'] = df['pickup_datetime'].dt.tz_convert('US/Eastern')  # Change the timezone to EST.
df['Hour'] = df['EDTDate'].dt.hour
df['AMorPM'] = np.where(df['Hour'] > 12, 'pm', 'am')
df['Weekday'] = df['EDTDate'].dt.strftime('%a')  # OR 'dt.dayofweek' to get the day of the week as a number.

print(df.head())

print("""
# #################################################################################################
#  Full ANN Code Along - Regression Part Two(2) (Categorical and Continuous Features)
# #################################################################################################
""")

categorical_cols = ['Hour', 'AMorPM', 'Weekday']
continuous_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                   'passenger_count', 'dist_km']
y_col = ['fare_amount']

# 1. CATEGORICAL COLUMNS - By mapping columns to 'category' data type, it allows to access alphanumeric data using
# numerical values. So the neural network can access those numerical data to train.

# Change the categorical columns's data type('dtype') to 'category' type, so neural network can understand the
# numerical codes from categories.
for col in categorical_cols:
    df[col] = df[col].astype('category')

print(df['Hour'].head())
print(df['AMorPM'].head())
print(df['Weekday'].head())

# We can access 'category' type object details using '.cat' as in '.dt' for 'DateTime' objects.
print(df['AMorPM'].cat.categories)
print(df['AMorPM'].cat.codes)

vals = df['AMorPM'].cat.codes.values    # Return a numpy array, so we can convert into a tensor.
print(type(vals))
print(vals)

# Derive codes from categorical columns and stack them up as single numpy array.
hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkd = df['Weekday'].cat.codes.values

cats = np.stack([hr, ampm, wkd], axis=1)  # Joint each array as a column , thus axis=1.
print(cats)

# #####################################################################################################################
# Use 'list' comprehension to derive values and stack them up to single numpy array. Only need following three lines.
# Summarize above steps using list comprehensions.
# #####################################################################################################################
categorical_cols = ['Hour', 'AMorPM', 'Weekday']
for col in categorical_cols:
    df[col] = df[col].astype('category')

cats = np.stack([df[col].cat.codes.values for col in categorical_cols], axis=1)
print(cats)

# OR just reduce the 'for loop for change data type to category' as well. ↓

cats = np.stack([df[col].astype('category').cat.codes.values for col in categorical_cols], axis=1)
print(cats)

# #####################################################################################################################
# Side effect here is, it doesnt change the original DataFrames type, but extract the mutated columns as we want. ↑
# #####################################################################################################################

# CONVERT stacked numpy array to 'tensor'
cats = torch.tensor(cats, dtype=torch.int64)

# 2. CONTINUOUS COLUMNS - Simply map them into a numpy since they are already numerical values and neurons basicaly
# understand them.

conts = np.stack([df[col].values for col in continuous_cols], axis=1)
print(conts)

# CONVERT stacked numpy array to 'tensor'
conts = torch.tensor(conts, dtype=torch.float)

# CREATE the labels using 'fare_amount' columns, hence need to predict 'fare_amount' base on categorical and continuous
# columns values after training.

# Reshape to have single columns, multiple rows.
y = torch.tensor(df['fare_amount'].values, dtype=torch.float).reshape(-1, 1)
print(y)

# After all the data prepared.
print(cats.shape)
print(conts.shape)
print(y.shape)

# #################################################################################################
# SET EMBEDDING Sizes
# #################################################################################################
# 1. Step - Take category sizes.
cat_szs = [len(df[col].cat.categories) for col in categorical_cols]
print(cat_szs)

# 2. Step - Take category sizes.
embedding_szs = [(szs, min(50, (szs + 1) // 2)) for szs in cat_szs]
print(embedding_szs)

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################

print("""
# ##############################################################################################################
#  Full ANN Code Along - Regression Part Three(3) (Tabular Model - Embedding, Normalization, Dropout functions)
# ##############################################################################################################
""")

cats_part = cats[:4]
print(cats_part)

# Embedding for categorical data.
self_embeds = nn.ModuleList([nn.Embedding(vocab_szs, vector_szs) for vocab_szs, vector_szs in embedding_szs])
print(self_embeds)

# FORWARD METHOD (cats)
embeddings = []
for i, e in enumerate(self_embeds):
    embeddings.append(e(cats_part[:, i]))

print(embeddings)

# Concatenate embeddings together into one tensor.
z = torch.cat(embeddings, 1)
print(z)

# Dropout some values from dense-vector to avoid over-fitting.
self_embed_dropout = nn.Dropout(p=0.4)
z = self_embed_dropout(z)
# Could see in the output some some values in the tensor zero(0) out randomly.
print(z)


# Build the 'TabularModel' class.
class TabularModel(nn.Module):

    def __init__(self, embedding_size, num_continuous_features, output_size, layers, p=0.5):
        super().__init__()
        # Map to a 'nn.ModuleList'.
        self.embeds = nn.ModuleList([nn.Embedding(vocab_szs, vector_szs) for vocab_szs, vector_szs in embedding_size])
        # Define 'nn.DropOut' with percentage to dropout.
        self.embed_dropout = nn.Dropout(p)
        # Define 'nn.BatchNorm1d(int)' with number of distinct continuous features.
        self.normalized_continuous_data = nn.BatchNorm1d(num_continuous_features)

        layer_list = []
        # Get the number of embedded features for categorical features (Essentially the sum of vector size of the
        # each of embeddings).
        num_embed = sum([vector_szs for vocab_szs, vector_szs in embedding_size])
        # Sum categorical/embedded features count and continuous features count to get input layer size.
        num_in = num_embed + num_continuous_features

        # Append each of the layers into a single list.
        for v in layers:
            layer_list.append(nn.Linear(num_in, v))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(v))
            layer_list.append(nn.Dropout(p))
            num_in = v

        # Append the last/output layer.
        # Output size should be 1 hence this is a regression problem and guessing a single value.
        layer_list.append(nn.Linear(layers[-1], output_size))

        # Flat out each of the layers and bind them sequentially.
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x_cat, x_cont):
        _embeddings = []

        # Send categorical features through the embeddings and fill the vectors of 'nn.Embedding' with random numbers
        # distinct to each of the feature's values.
        for v, emb in enumerate(self.embeds):
            _embeddings.append(emb(x_cat[:, v]))

        # Concatenate each of the embedding
        x = torch.cat(_embeddings, 1)
        x = self.embed_dropout(x)

        x_cont = self.normalized_continuous_data(x_cont)
        x = torch.cat([x, x_cont], 1)

        x = self.layers(x)
        return x


# What does 'nn.Sequential' DO?
s = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 10))
print(s)

inpt = torch.linspace(0, 10, 10)
outpt = s(inpt)
print(outpt)

print("""
# #################################################################################################
#  Full ANN Code Along - Regression Part Four(4) 
# #################################################################################################
""")

torch.manual_seed(33)

model = TabularModel(embedding_szs, conts.shape[1], 1, [200, 100], p=0.4)
print(model)

# Train the model with available data and train,test split.
criterion = nn.MSELoss() # np.sqrt(MSE) -> RMSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_batch_size = 60000
test_batch_size = int(60000 * 0.2)   # 20% of training size.

# Train, test split. (DATA ALREADY SHUFFLED)
cat_train = cats[:train_batch_size - test_batch_size]
cat_test = cats[train_batch_size - test_batch_size:train_batch_size]
cont_train = conts[:train_batch_size - test_batch_size]
cont_test = conts[train_batch_size - test_batch_size:train_batch_size]

y_train = y[:train_batch_size - test_batch_size]
y_test = y[train_batch_size - test_batch_size : train_batch_size]

print(len(cat_train))
print(len(cont_train))
print(len(y_train))
print(len(cat_test))
print(len(cont_test))
print(len(y_test))

# Train the model with fix EPOCHS.
start = time.time()

epochs = 150
losses = []

for i in range(epochs):

    y_pred = model.forward(cat_train, cont_train)
    # Calculate Root Mean Square Error hence dealing with price units, otherwise loss values
    # (in this case 'fare_amount') would be squared.
    loss = torch.sqrt(criterion(y_pred, y_train))

    losses.append(loss)

    if i % 10 == 1:
        print(f'epoch {i} -> loss: {loss}')

    # Back propagate.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

duration = time.time() - start
print(f'Training took {duration / 60} minutes')

# Plot learning curve.
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('RMSE Loss')
plt.show()

# Validate the model on '*test' sets.
with torch.no_grad():

    y_pred = model.forward(cat_test, cont_test)

    loss = torch.sqrt(criterion(y_pred, y_test))

    print(loss)     # This loss should be around the last epochs loss value, this to be well fitted dataset.


# Print out the predicted, actual and different of the values ('fare_amount').
for i in range(10):

    diff = np.abs(y_pred[i].item() - y_test[i].item())
    print(f'{i}.) PREDICTED: {y_pred[i].item():8:2f}, ACTUAL: {y_test[i].item():8.2f}, DIFF: {diff}')
