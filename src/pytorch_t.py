#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

print('''
# #################################################################################################
# PyTorch Basics
#
#   - PyTorch Tensors
#       * Basics
#       * Creating Tensors
#       * Operations
#       * Exercise 
# 
#   - Tensors - A general representation of Vector and Matrix.
#       * Scalar (Not a Tensor)
#       * 1D Tensor - Vector
#       * 2D Tensor - Matrix
#       * 3D Tensor
#       * ND Tensor (4D, 5D and higher dimensional Tensors) 
#        
# Tensor Basic - Part One
# #################################################################################################
''')

arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(arr.dtype)
print(type(arr), end='\n\n')

x = torch.from_numpy(arr)   # Retain the link with NumPy array.
print(x)
print(x.dtype)
print(type(x), end='\n\n')

x = torch.as_tensor(arr)    # Retains the link with NumPy array.
                            # Same as the preceding statement 'torch.from_numpy(arr)'.

arr[0] = 99
print(arr)
print(x)

# Changes done to numpy array also reflects in the tensor 'x' object created with
# 'tensor.as_tensor(arr)' and 'tensor.from_numpy(arr)'.

arr_2d = np.arange(0., 12.)
print(arr_2d)

arr_2d = arr_2d.reshape(4, 3)
print(arr_2d)

x2 = torch.as_tensor(arr_2d)
print(x2)

# Above statements also create a tensor object which reflects changes do to numpy array.
# To avoid this use just 'tensor.tensor(data)' function.

x2 = torch.tensor(arr_2d)
print(x2)

print('''
# #################################################################################################
# Tensor Basics - Part Two
# #################################################################################################
''')

# Different between 'torch.tensor(data)', 'torch.Tensor(data)' and 'torch.FloatTensor(data)' functions.
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr)
print(arr.dtype) # dtype = int32

t1 = torch.tensor(arr)
print(t1)
print(t1.dtype)  # dtype = int32 (Retained the original numpy arrays data type)

t2 = torch.Tensor(arr)
print(t2)
print(t2.dtype)  # dtype = float32 (Change data type to floating point)

t3 = torch.FloatTensor(arr)
print(t3)
print(t3.dtype)  # dtype = float32 (Change date type to floating point)

# torch.Tensor and torch.FloatTensor are classes just like torch.BooleanTensor and other wrapper type tensors.
# But 'torch.Tensor(data)' is an alias for 'torch.FloatTensor(data)'. They are the exact same.

# INITIALIZE TENSORS from the scratch.

x = torch.empty(4, 3)   # Create a placeholder until data fills. Not actually empty hence this returns
                        # data in current memory location even though do not assign data explicitly.
print(x)

x = torch.zeros(4, 3)   # Create a tensor with actual zero(0) values. Default data type is float.
print(x)

x = torch.zeros(4, 3, dtype=torch.int16)    # Create tensor with manually assigned data type.
print(x)

x = torch.ones(4, 3)    # Create a tensor with actual one(1) values.
print(x)

# Torch functions like NumPy.

x = torch.arange(0, 20, 2)
print(x)

x = torch.arange(0, 18, 2).reshape((3, 3))
print(x)

x = torch.linspace(0, 18, 12).reshape(4, 3)
print(x)

# Tensor can be created with Python list as well.
x = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(2, 3)
print(x)

# Change the data type of already created tensor.
x = torch.tensor([1, 2, 3, 4])
print(x.dtype)
print(x)

x = x.type(torch.int16)
print(x.dtype)
print(x)

# Working with random numbers.
x = torch.rand(4, 3)     # Create random number of uniform distributions between 0 and 1.
print(x)

x = torch.randn(4, 3)   # Create random number of standard normal distributions.
                        # (mean = 0, standard deviation(sigma) = 1)

i = torch.randint(0, 10, (5, 5))    # Size tuple is mandatory.
print(i)

# Create tensors like(size/dimensions) other tensors.
x = torch.zeros(4, 3)
print(x)
print(x.shape)

n1 = torch.rand_like(x)     # Automatically grab the "SHAPE" of the given tensor and create
                            # a new tensor with uniform distributions.
print(n1)

n2 = torch.randn_like(x)    # Create a new tensor with normal distribution with the shape of 'x'.
print(n2)

# New tensor with random numbers with the shape of 'x'.
n3 = torch.randint_like(x, low=0, high=100, dtype=torch.int32)
print(n3)

# Setup a seed for random number generation.
torch.manual_seed(42)   # 42 because of the reference to hitchhiker's guide to the galaxy.

x = torch.rand(2, 3)
print(x)

# Execution device.
print('Device: ', x.device)
print('Size: ', x.size())   # x.shape and x.size() are the same.
print('Layout: ', x.layout)

print('''
# #################################################################################################
# Tensor Operations - Part One
# #################################################################################################
''')

x = torch.arange(6).reshape(3, 2)
print(x)

val = x[1, 1]
print(val)      # Still return a Tensor object wrapping the number/s.
print(type(val))

# Slicing tensors.
val = x[:, 1]   # This doesn't retain the structure of returning column.
print(val)

val = x[:, 1:]  # This retains the column structure hence column slicing instead of specific index number.
print(x)

# 't.reshape(int, int)' vs 't.view(int, int)' functions.
x = torch.arange(1, 10)
print(x)

reshaped = x.reshape(3, 3)
viewed = x.view(3, 3)

print(reshaped)
print(viewed)

print(x)    # t.view() and t.reshape() functions are essentially evenly function by returning reoriented tensor.

reshaped[0, 1] = 999
viewed[1, 1] = 999

print(x)
# Both the function retain the link with the original tensor and any change do to the viewed or reshaped tensor
# would alter the original tensor.
#
# By changing the original tensor values, we can reflects changes on reshaped and viewed tensors.
x[7] = 99999

print(x)
print(reshaped)
print(viewed)

# view() can infer the dimension by looking at other dimension value an reshape the tensor.

x = torch.arange(0, 10)
print(x)

print(x.view(2, -1))    # Here returns a (2, 5) tensor hence -1 infer the other dimension.
print(x.view(-1, 5))    # On either dimensions it works.

# Adopt another tensor's shape with '.view_as(tensor)'
x = torch.arange(1, 10)
print(x)

x = x.view_as(reshaped)  # 'reshaped' is a (3, 3) tesnsor which created earlier in the file.
print(x)

# Basic arithmetic with Tensor objects.

a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

print(a, b)

print(a + b)
print(torch.add(a, b))

# Inplace change arthmatic functions.

a = a.add(10)   # Add whatever the value to the caller tensor and return. Do not change inplace.
print(a)

a = a.add(b)    # Same here. Add the tensor to caller tensor and return, don't change the tensore
                # inplace unless reassign.
print(a)

a.add_(b)   # Add in place. No need reassign. Almost all the arithmetic functions has these underscored
            # (_) methods and they all do change the caller tensor in place.
print(a)

a.mul_(2)   # Change in place.
print(a)

print(a.sum())  # Supplementary operation.
print(a.mean())
print(a.std())
print(a.var())
print(a.median())
# On module functions.
print(torch.sqrt(a))
# Trigonometric operations
print(torch.sin(a))
print(torch.cos(a))

print('''
# #################################################################################################
# Tensor Operations - Part Two (Matrix Multiplication, Dot Products, Advance Functions)
# #################################################################################################
''')

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Dot products. All three ways belows are the same operation. Should match dimensions of two tensors by size.
print(torch.dot(a, b))
print(a.dot(b))
print(b.dot(a))

# If tensor shapes are not similar (no matter orientation is different), dot operation would fail.
c = torch.tensor([7, 8, 9, 10])
# print(a.dot(c))  # RuntimeError: inconsistent tensor size, expected tensor [3] and src [4] to have
# the same number of elements, but got 3 and 4 elements respectively

# Matrix multiplication.
a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(a)
print(b)

print(torch.mm(a, b))   # Matrix multiplication.
print(a.mm(b))
print(b.mm(a))

print(a @ b)  # Also works. But '@' use to define python decorators. So, '.mm(t)' function recommended.

# L2 and Euclidian Norm
x = torch.tensor([2., 3., 4., 5.])

print(x.norm())     # Euclidian Norm
print(x.numel())    # Number of elements in the tensor.
print(len(x))       # Same for 1D tensors.

print(len(a))       # Only return number of rows. So must use '.numel()' function.
print(a.numel())
