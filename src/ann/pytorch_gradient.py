#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

x = torch.tensor(2., requires_grad=True)
print(x)

y = 2 * x ** 4 + x ** 3 + 3 * x ** 2 + 5 * x + 1
print(y)

print(type(y))

y.backward()    # Back propagate the calculations done to 'x'.
print(x.grad)

# Back propagate through multiple steps (complex).
x = torch.tensor([[1., 2., 3.], [3., 2., 1.]], requires_grad=True)
print(x)

y = 3 * x + 2
print(y)

z = 2 * y ** 2
print(z)

out = z.mean()
print(out)

out.backward()
print(x.grad)

# If do not want the gradient tracking.
x.requires_grad_(False)
# OR
x.detach()
# OR skip tracking for specific calculations 'torch.no_grad(x)' avoid tracking.
