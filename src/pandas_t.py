#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from numpy.random import randn

print('''
# #################################################################################################
# Pandas - Abbreviation of "Panel Data"
# 
# Panda Series
# ################################################################################################
''')

labels = ['a', 'b', 'c']

my_list = [10, 20, 30]

arr = np.array(my_list)
print(arr)

# Using a Python list.
p = pd.Series(my_list)
print(p)

# Using a NumPy array.
p = pd.Series(data=arr, index=labels)
print(p)

# How to access Series.
ser1 = pd.Series([1, 2, 3, 4], index=['USA', 'Germany', 'USSR', 'Japan'])
print(ser1)

print(ser1['USA'])
print(ser1['Japan'])

ser2 = pd.Series([1, 4, 5, 6], index=['USA', 'Germany', 'Italy', 'Japan'])

ser3 = ser1 + ser2
print(ser3)

'''
Germany     6.0     # All the numbers turned to float. (Sake of precision) 
Italy       NaN     # 'ser1' has no 'Italy' index, thus, arithmetic operation of it's value happens with nothing, 
                    # so it's result NaN (Not a number).
Japan      10.0     
USA         2.0
USSR        NaN     # Same happens to 'USSR' due to unavailability of index in 'ser2'.
dtype: float64
'''

print('''
# #################################################################################################
# Data Frames - Part One
# #################################################################################################
''')

np.random.seed(101)
rand_mat = randn(5, 4)
print(rand_mat, end='\n\n')

df = pd.DataFrame(data=rand_mat, index='A B C D E'.split(), columns='W X Y Z'.split())
print(df, end='\n\n')

# Select single COLUMN.
colW = df['W']
print(type(colW))
print(colW, end='\n\n')

# OR

print(df.W, end='\n\n')     # Not recommended.

# Select multiple COLUMNS.
colWY = df[['W', 'Y']]
print(colWY, end='\n\n')

print(df[['Z', 'X']], end='\n\n')

# Create a new COLUMN.
df['NEW'] = df['Y'] + df['X']
print(df, end='\n\n')

# Remove/drop column or row.
df.drop('NEW', axis=1, inplace=True)    # KeyError: "['NEW'] not found in axis", unless axis provided.
                                        # Thus, either columns of rows can be dropped.
                                        # Set 'inplace=True' to affect the DataFrame without reassigning.
print(df, end='\n\n')

print(df.drop('A', axis=0), end='\n\n') # Not a 'inplace' delete.

# Selecting single ROWS.
print(df.loc['A'], end='\n\n')  # By index name string.
print(df.iloc[2], end='\n\n')   # By numerical index, like an array.

# Selecting multiple ROWS.
print(df.loc[['A', 'C']], end='\n\n')
# OR
print(df.iloc[[1, 3, 0]], end='\n\n')

# Select multiple ROWS', specific columns.
print(df.loc[['A', 'C']][['X', 'Z']], end='\n\n')
# OR
print(df.iloc[[2, 1]][['W', 'Z']], end='\n\n')
# OR
print(df.loc[['A', 'B'], ['W', 'X']], end='\n\n')
# OR
print(df.iloc[[1, 2, 3], [0, 1]], end='\n\n')

'''
COLUMNS CANNOT ACCESS USING INDEX NUMBERS UNLESS THEY ARE IN FIRST SQUARE BRACES ([]) !!!
'''

print('''
# #################################################################################################
# Data Frames - Part 2
# #################################################################################################
''')

matrix = randn(5, 4)

df = pd.DataFrame(data=matrix, index='A B C D E'.split(), columns='W X Y Z'.split())
print(df, end='\n\n')

# Conditional DataFrames.
bool_df = df > 0
print(bool_df)

print(df[bool_df], end='\n\n')

# OR shorter

print(df[df > 0], end='\n\n')

# Filter by specific column values.
print(df[df['W'] > 0], end='\n\n')
# Filter by specific columns values and get specific columns.
print(df[df['X'] > 0][['W', 'Z']], end='\n\n')
# Filter by specific column values and get specific cell value.
print(df[df['Y'] > 0]['Z'].iloc[0], end='\n\n')
# SINCE 'A' COLUMN NOT AVAILABLE IN PRECEDING OUTPUT USING INDEX LOCATION '0' WOULD RETURN IMMEDIATE
# MEMBER OF THE TABLE. IN THIS CASE IT'S 'B' COLUMN.

# Multiple conditions on DataFrames
cond1 = df['W'] > 0
cond2 = df['Y'] > 0

# 'df[cond1 and cond2]' # Python inbuilt 'and' and 'or' keywords doesn't work with Pandas.
# Instead use &(ampersand) and |(pipe) operators.

print(df[cond1 & cond2], end='\n\n')

# OR shorter.

print(df[(df['W'] > 0) & (df['Y'] > 0)], end='\n\n')  # Use braces to wrap the condition is kind of mandatory.

# More with DataFrame indexes.
print(df.reset_index(inplace=False), end='\n\n')      # Change the indexes. Default not inplace (False).

# Update index with new set of index names (Resetting and setting indices).
new_idxs = 'CA NY WY OR CO'.split()
df['States'] = new_idxs  # Adding new columns.

print(df, end='\n\n')

df.set_index('States', inplace=True)   # Say specific column to be the index.
print(df, end='\n\n')

# Get summaries from DataFrames.
print(df.info(), end='\n\n')
print(df.dtypes, end='\n\n')
print(df.describe(), end='\n\n')
print(df.sum(axis=1), end='\n\n')

# Function of series derived from DataFrames.
ser_w = df['W'] > 0

print(ser_w)
print(ser_w.value_counts())
print(sum(ser_w))
print(len(ser_w))

print('''
# #################################################################################################
# Missing Data
# #################################################################################################
''')
df = pd.DataFrame({'A': [1, 2, np.nan],
                   'B': [5, np.nan, np.nan],
                   'C': [1, 2, 3]})
print(df, end='\n\n')

print(df.dropna(inplace=False), end='\n\n')     # Drop ROWS with 'NaN' object/s.
print(df.dropna(axis=1), end='\n\n')            # Drop COLUMNS with 'NaN' object/s (Across axis=1).
print(df.dropna(thresh=2), end='\n\n')          # Drop ROWS with more than 2 'NaN' objects.
print(df.dropna(axis=1, thresh=2), end='\n\n')  # Drop COLUMNS with more than 2 'NaN' objects.

print(df.fillna('NEW'), end='\n\n')     # Fill 'NaN' objects with given value.

df['A'].fillna(value=df['C'].mean(), inplace=True) # Fill 'NaN' objects in 'A' columns with 'C' columns mean value.
print(df, end='\n\n')

print(df['B'].mean())   # Still works eve though there are 'NaN' object in the 'B' row, ignoring them.

print('''
# #################################################################################################
# Group By Operations
# #################################################################################################
''')

df = pd.DataFrame({'Company': ['GOOG', 'GOOG',  'MSFT', 'MSFT', 'FB', 'FB'],
                   'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
                   'Sales': [200, 120, 340, 124, 243, 350]})
print(df, end='\n\n')

group = df.groupby('Company')
print(group.describe())
print(group.count())
print(group.mean())
print(group.sum())
print(group.median())
print(group.describe().transpose())  # Rotate the table to switch ROW and COLUMN locations.

print('''
# #################################################################################################
# Pandas Operations (Extra methods and attributes)
# #################################################################################################
''')

df = pd.DataFrame({'col1': [1, 2, 3, 4],
                   'col2': [444, 555, 666, 444],
                   'col3': ['abc', 'def', 'ghi', 'jkl']})

print(df, end='\n\n')

print(df['col2'].unique())                  # Unique values.
print(df['col2'].nunique(dropna=False))     # Number of unique values.
print(df['col2'].value_counts())            # Number of occurrences of the values.

# Create DF col1 > 2 and col2 == 444
new_df = df[(df['col1'] > 2) & (df['col2'] == 444)]
print(f'\n{new_df}\n')

# Apply new values to every single values in column or table. 'series.map(funcs)' also the same.
df['NEW'] = df['col1'].apply(lambda num: num * 2)
print(df)

del df['NEW']       # Delete column.
print(df)

df['NEW'] = df['col1'].map(lambda num: 'EVEN' if num % 2 == 0 else 'ODD')
print(df)

df.drop('NEW', axis=1)
print(df)

# ANOTHER USE CASE OF 'map()' function passing a dictionary.
#   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html
df['NEW'] = df['col1'].map({1: 'ONE', 2: 'TWO', 3: 'THREE'})
print(df)

df.drop('NEW', axis=1)
print(df)

print(df.columns)   # Column names.
print(df.index)     # Index names.
print(df.info())

df.sort_values('col2', axis=0, inplace=True, ascending=False)   # Sort values.
print(df)

print(df['col1'].idxmin())  # Get the minimum value's index.
print(df['col1'].idxmax())  # Get the maximum value's index.

print('''
# #################################################################################################
# Pandas Data Input and Output - https://pandas.pydata.org/pandas-docs/version/0.22/io.html
# #################################################################################################
''')

# Read CSV and Save As CSV
csv = pd.read_csv('../notebooks/00-Crash-Course-Topics/01-Crash-Course-Pandas/example.csv')
print(csv)

new_df = csv[['a', 'b']]
print(new_df)

new_df.to_csv(path_or_buf='../resources/output/example_out1.csv', index=False)  # Exclude default indexes.

# Read Excel and Save As CSV
excel = pd.read_excel('../notebooks/00-Crash-Course-Topics/01-Crash-Course-Pandas/Excel_Sample.xlsx',
                      sheet_name='Sheet1')
excel.drop('Unnamed: 0', axis=1)
print(excel)


new_df = excel[['a', 'b', 'd']]
print(new_df)

new_df.to_csv('../resources/output/example_out2.csv', index=False)

# Read from HTML.
html = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
print(html)
print(type(html))   # Is a list.
print(len(html))

df = html[0]
print(df.info(), end='\n\n')
print(len(df['City']))

df.to_csv('../resources/output/example_out3.csv', index=False)

# #################################################################################################
# Notes :-
#
#   1. Statement 'len(df)' returns how many rows/index are available in the DataFrame.
#   2. Statement 'len(df["col1"])' also return the length of the Series essentially.
#      It looks like 'len(Series)'
#
# ################################################################################################
