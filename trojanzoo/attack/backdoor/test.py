import torch

a = torch.rand((4, 3, 3))
mask = torch.tensor([[0, 0, 0],[0, 1, 1],[0, 1, 1],])
mark = torch.tensor([[-1, -1, -1],[-1, -1, -1],[-1, -1, -1]])

a = a * (1-mask) + mask * mark

print(a)