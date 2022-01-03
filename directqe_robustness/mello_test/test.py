import torch
from torch.nn import functional as F
import copy
import json

a = torch.tensor([[1, 2], [3, 4]])
print(a.size())
b =  a.repeat(1, 2, 1, 1)
print(b.size())
print(b)


# python mello_test/test.py