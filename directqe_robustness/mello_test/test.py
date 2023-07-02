import torch

a = torch.tensor([[1, 2, 0], [4, 0, 0]])
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
mask = (a != 0)
print(a)
print(b * mask)

# python mello_test/test.py