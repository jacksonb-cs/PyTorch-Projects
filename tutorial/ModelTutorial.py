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

		x = self.flatten(x)
		logits = self.linear_relu_stack(x)

		return logits


# ========== Main Program ========== #

model = NeuralNetwork().to(device)
# print(model)

in_data = torch.rand(1, 28, 28, device=device)
logits = model(in_data)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f'Predicted class: {y_pred}')

# TO-DO: Continue at "Model Layers"
