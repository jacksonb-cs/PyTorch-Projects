from textwrap import indent
import torch
import numpy as np

in_data: list = [[2, 3], [1, 4]]
data_t: torch.Tensor = torch.tensor(in_data)

data_ones: torch.Tensor = torch.ones_like(data_t)
# print(f"Ones Tensor:\n{data_ones}\n")

data_rand: torch.Tensor = torch.rand_like(data_t, dtype=torch.float16)
# print(f"Random Tensor:\n{data_rand}\n")

shape: tuple = (3, 3,)
rand_tensor: torch.Tensor = torch.rand(shape)
ones_tensor: torch.Tensor = torch.ones(shape)
zeros_tensor: torch.Tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

# print(f"Shape of tensor: {data_t.shape}")
# print(f"Datatype of tensor: {data_t.dtype}")
# print(f"Device tensor is stored on: {data_t.device}")

if torch.cuda.is_available():
	data_td = data_t.to('cuda')

# print(f"Tensor is on {data_td.device}.")

tensor = torch.ones(4, 4)
# print('First row: ', tensor[0])
# print('First column: ', tensor[:, 0])
# print('Last column: ', tensor[..., -1])
tensor[:, 1] = 0
tensor[..., -1] = 2
# print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

# Standard matrix multiplication
# y1, y2, and y3 are all the same thing
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# Element-wise multiplication
# z1, z2, and z3 are all the same thing
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

aggregate = tensor.sum()
agg_item = aggregate.item()
print(agg_item, type(agg_item))

print("\n===== DONE =====\n")
