# All TorchVision datasets have two parameters:
# 	- transform: modify the features
# 	- target_transform: modify the labels
#
# For example, FashionMNIST features are in PIL Image format and labels
# are integers. In order to train over this data, we need to convert
# the features into normalized tensors and the labels as one-hot
# encoded tensors.
# We do this using ToTensor and Lambda

from turtle import down
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
	root='data',
	train=True,
	download=True,
	transform=ToTensor(),
	target_transform=Lambda(
		lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
	)
)

print('\n========== DONE ==========')
