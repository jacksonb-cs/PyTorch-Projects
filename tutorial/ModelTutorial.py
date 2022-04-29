import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Every module in PyTorch subclasses nn.Module
# A neural net is a module itself that consists of other modules (layers)
class NeuralNetwork(nn.Module):

	def __init__(self):

		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10),
		)

	def forward(self, x):

		x = self.flatten(x)		# Converts pixels to contiguous 1-D array
		logits = self.linear_relu_stack(x)

		return logits


# ========== Main Program ========== #

model = NeuralNetwork().to(device)
# print(model)

in_data = torch.rand(1, 28, 28, device=device)
logits = model(in_data)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
# print(f'Predicted class: {y_pred}')

# ===== Demo of layers doing their thang ===== #

# Three 28x28 "images"
input_image = torch.rand(3, 28, 28)
print(f'Input size: {input_image.size()}')

# Flatten 28x28 pixels to 1-D array (784 pixels)
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(f'Flattened img size: {flat_image.size()}')

# Linear layer applies a linear transformation (intuitive weights)
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(f'Hidden layer size: {hidden1.size()}\n')

# Non-linear functions are necessary to create complex mappings
# Here, we use ReLU
# print(f'Before ReLU: {hidden1}\n\n')
hidden1 = nn.ReLU()(hidden1)
# print(f'After ReLU: {hidden1}\n\n')

# The sequential module is a container of modules
# It executes modules in the order given
seq_modules = nn.Sequential(
	flatten,
	layer1,
	nn.ReLU(),
	nn.Linear(20, 10)
)
logits = seq_modules(input_image)

# We use softmax after the last layer to scale values to [0, 1].
# This represents probabilities an image belongs to each class.
# The dim parameter indicates the dimension along which the values
# should sum to 1.
softmax = nn.Softmax(dim=1)
pred_prob = softmax(logits)

print(f'Model structure: {model}\n')
for name, param in model.named_parameters():
	print(f'Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n')

print('\n========== DONE ==========')
