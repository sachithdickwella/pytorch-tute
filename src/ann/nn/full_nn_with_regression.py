#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

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

y = torch.tensor(df['fare_amount'].values).reshape(-1, 1)  # Reshape to have single columns, multiple rows.
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
embedded_szs = [(szs, min(50, (szs + 1) // 2)) for szs in cat_szs]
print(embedded_szs)

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################

print("""
# #################################################################################################
#  Full ANN Code Along - Regression Part Three(3) (Tabular Model)
# #################################################################################################
""")

