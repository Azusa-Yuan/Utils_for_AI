n = 1_00
print(n)
import torch
import torch.nn as nn
t = torch.tensor(1)
ti = torch.tensor([])
print(t)
print(ti)
# if is_master():
#     print(1)

w = torch.empty(3, 5)
nn.init.uniform_(w)
print(w)