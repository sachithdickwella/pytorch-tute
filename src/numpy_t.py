#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# NumPy is numerical processing library that can handle large data sets stored as arrays.

import numpy as np

print('''
# #################################################################################################
# NumPy Arrays - Part 1
# #################################################################################################
''')

# NumPy Arrays
my_list = [1, 2, 3]

ar = np.array(my_list)
print(type(ar))
print(ar)

my_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(my_list)

# Matrix/2D array creation with given nested lists.
my_matrix = np.array(my_list)
print(my_matrix)

# Dimension of the matrix as a tuple.
print(my_matrix.shape)

# Equivalent to range() function in python.
r = np.arange(0, 10)
print(r)

# OR with steps
r = np.arange(0, 10, 2)
print(r)

# Create array with zeros.
z = np.zeros(5)
print(z)  # With five(5) zeros in floating point.

# OR with initial shape for matrix
z = np.zeros((4, 10))   # 4 x 10 (rows x columns) matrix.
print(z)

# Create array with ones.
o = np.ones(10)
print(o)

# OR with initial shape for matrix.
o = np.ones((5, 5))
print(o)

# Broadcast a initial value across an array.
o = np.ones((10, 10)) + 20
print(o)

o = np.ones((5, 5)) + 10
print(o)

o = np.ones((6, 5)) / 100
print(o)

# Array with linearly spaced numbers. Given number of linearly spaced numbers between start and end.
ls = np.linspace(0, 10, 2)
print(ls)
print(len(ls))

ls = np.linspace(0, 10, 20)
print(ls)

# Identity matrix (1s across the diagonal and everything else are 0s).
e = np.eye(5)
print(e)

print('''
# #################################################################################################
# NumPy Arrays - Part 2
# #################################################################################################
''')

# Array with random numbers.
r = np.random.rand(5)
print(r)
print(r.shape)

# Matrix with random numbers.
r = np.random.rand(5, 5)
print(r)
print(r.shape)

# Array and matrix with standard distribution mean 0.
r = np.random.randn(1)
print(r)    # Random number deviate from 0.

r = np.random.randn(10)
print(r)

r = np.random.randn(6, 5)
print(r)

# Array and matrix with standard distribution mean custom.
r = np.random.normal(loc=5., scale=1, size=5)
print(r)

# One random number.
r = np.random.randint(0, 10)
print(r)

# Array of random number.
r = np.random.randint(0, 100, 10) # Array siz is 10 elements.
print(r)

# Random numbers with a seed.
np.random.seed(101)
r = np.random.rand(5)   # Return same random numbers despite how many time this statement executed hence the seed.
print(r)

# Arrays attributes and methods/behaviours.
arr = np.arange(25)
print(arr)

randarr = np.random.randint(0, 50, 10)
print(randarr)

# 'arr.reshape(tuple)' 1D to 2D array.
print(arr.shape)

arr = arr.reshape(5, 5)
print(arr)
print(arr.shape)

# arr.max() and arr.min()
r = np.random.randint(0, 100, 10)
print(r)

print(r.max())
print(r.min())

# OR get the index of max and min values.

print(r.argmax())
print(r.argmin())

# Get the data type of array elements.
print(r.dtype)

i = np.array([1, 2, 3, 'a', 'b', 'c', 4])
print(i)
print(i.dtype)  # 11 character long Unicode (U11).

print('''
# #################################################################################################
# Numpy Index Selection
# #################################################################################################
''')

arr = np.arange(0, 11)
print(arr)

print(arr[8])
print(arr[1:5])
print(arr[5:])
print(arr[:4])
print(arr[::-1])

print(arr * 2)
print(arr / 3)
print(arr ** 2)

print(arr)  # No change to the original array.

slice_of_array = arr[:6]
print(slice_of_array)

slice_of_array[:] = 99
print(slice_of_array)   # This has been changed.
print(arr)              # First five(5) elements also have been changed.
# Even though sliced, sliced array still pointed to original array.
# To avoid this use 'arr.copy()' function to create distinct copy in memory.

arr = np.arange(0, 11)

slice_of_array = arr[:6].copy()
print(slice_of_array)

slice_of_array[:] = 99
print(slice_of_array)
print(arr)

# INDEXING on 2D array/Matrix.
arr_2d = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
print(arr_2d)
print(arr_2d.shape)

print(arr_2d[1][1])
# OR
print(arr_2d[1, 1])  # Same as 'arr[idx1][idx2]' if only pick single element.

print(arr_2d[-1, -1]) # Negative indexes work.

print('''
# In 2D arrays/Matrix slicing the output can be different;
#
# Ex:-
#   Between the usage of array[x:, y:] and array[x:][y:]) can be totally
#   different due to their execution order. Should be overly consider while
#   dealing with them. It's even further getting complex when use with steps.
# 
# Below:
''')

# 2D array SLICING.
print(arr_2d[:2, 1:])   # Retain the structure. Take values at once.

# Straighten up the 2d array to 1d array. Execute first braces first,
# then second braces execute on first one's results.
print(arr_2d[:2][:, 1:])  # Similar output with preceding statement.

# Slicing out a specific column.
print(arr_2d[:, 1])

# Slicing with steps.
print(arr_2d[::-1, ::-1])

print(arr_2d[1::2, ::-2])

print('\n# Conditional SELECTION\n')

arr = np.arange(1, 11)
print(arr)

bool_arr = arr > 4
print(bool_arr)

print(arr[bool_arr])    # Filter out matching index elements from 'arr' array which has 'False' in 'bool_arr'.

# In a nutshell, this is actual shorten syntax.
print(arr[arr > 4])

print('''
# #################################################################################################
# NumPy Operations
# #################################################################################################
''')

arr = np.arange(0, 10)

print(arr + 2)
print(arr - 2)
print(arr * 2)
print(arr / 2)
print(arr ** 2)
print((arr ** 2) / 100)
print(arr * arr)
# print(arr / arr)    # Warning: the operation at index 0, 0/0 (0 divided by 0).
# print(1 / arr)      # Warning: the operation at index 0, 1/0 (1 divided by 0).

# Universal mathematical operations.
arr = np.arange(1, 11)

print(np.sqrt(arr))
print(np.exp(arr))
print(np.log(arr))
print(np.sin(arr))  # Trigonometric functions.
print(np.cos(arr))
print(np.tan(arr))
print(np.tanh(arr))

# Array statistics.
print(arr.max())
print(arr.min())
print(arr.sum())
print(arr.mean())
print(arr.var())    # Variance
print(arr.std())    # Standard deviation

# Axis Logic
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(arr_2d)

# GIVE SUM Across the ROWS (axis=0)
print(arr_2d.sum(axis=0))

# GIVE SUM Across the COLUMNS (axis=1)
print(arr_2d.sum(axis=1))

# Exercise Question

print(np.arange(1, 50.5, 0.1))  # Start, end and step sizes can be floating point numbers.

'''
To retain the 2-dimensional structure, should slice the structure, 
rather than get the value from specific index number.
'''
