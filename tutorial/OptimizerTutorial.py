import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class FashionNet(nn.Module):

	def __init__(self):

		super(FashionNet, self).__init__()
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
		return self.linear_relu_stack(x)

def train_loop(dataloader: DataLoader, model, loss_fn, optimizer, device):

	size = len(dataloader.dataset)
	for batch, (x, y) in enumerate(dataloader):

		# Send data to appropriate device
		x, y = x.to(device), y.to(device)

		# Compute prediction and loss
		pred = model(x)
		loss = loss_fn(pred, y)

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:

			loss = loss.item()
			current = batch * len(x)
			print(f'Loss: {loss:.4f}\t[{current}/{size}]')

def test_loop(dataloader: DataLoader, model, loss_fn, device):
	
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct = 0, 0

	with torch.no_grad():
		for x, y in dataloader:

			x, y = x.to(device), y.to(device)

			pred = model(x)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()

	test_loss /= num_batches
	correct /= size
	print(f'Test error:\n  Accuracy: {(100*correct):.1f}%, Avg loss: {test_loss:.4f}\n')


training_data = datasets.FashionMNIST(
	root='data',
	train=True,
	download=True,
	transform=ToTensor()
)

test_data = datasets.FashionMNIST(
	root='data',
	train=False,
	download=True,
	transform=ToTensor()
)

# I messed with these a little bit but I'm not here to perfect it
learning_rate = 2e-3
epochs = 25
batch_size = 64
loss_fn = nn.CrossEntropyLoss()

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = FashionNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):

	print(f'Epoch {epoch + 1}\n-------------------------------')
	train_loop(train_dataloader, model, loss_fn, optimizer, device)
	test_loop(test_dataloader, model, loss_fn, device)

print('Done!')
