import os

import matplotlib.pyplot as plt

from skimage import io

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Lambda

class TankPics(Dataset):

	def __init__(self, img_dir, label_map: dict=None, train=True, transform=None, target_transform=None):
		
		self.img_dir = img_dir
		self.label_to_int = label_map
		self.train = train
		self.transform = transform
		self.target_transform = target_transform

		# TO-DO: Determine if making and storing this dict here is optimal
		# Other option is to do it in the target transform
		if not self.label_to_int:

			self.label_to_int = {
				'2s1': 0,
				'bmp2': 1,
				'btr70': 2,
				'm1': 3,
				'm2': 4,
				'm35': 5,
				'm60': 6,
				'm548': 7,
				't72': 8,
				'zsu23': 9,
			}
		
		self.file_names = []

		for _, _, files in os.walk(img_dir):
			
			self.file_names.extend(files)

		# TO-DO: Better way to do this? Move pics to different directories?
		# This is really ugly, but it's only done when the data is initially loaded, so...
		for file in reversed(self.file_names):
			
			suffix = file.split('Deg_0', 1)[1]	# Get second half of split
			degree = int(suffix.split('_', 1)[0])	# Get first half of split as integer

			if self.train and degree > 16:

				self.file_names.remove(file)

			elif not self.train and degree < 17:

				self.file_names.remove(file)

	def __len__(self):
		
		return len(self.file_names)

	# TO-DO: Preferable to return scalar label or one-hot tensor?
	def __getitem__(self, index):
		
		file_name = self.file_names[index]

		label_str = file_name.split('_', 1)[0]
		img_path = os.path.join(self.img_dir, label_str, file_name)

		image = io.imread(img_path)
		label = self.label_to_int[label_str]

		if self.transform:
			image = self.transform(image)

		if self.target_transform:
			label = self.target_transform(label)

		return image, label


class TankNet(nn.Module):

	def __init__(self):

		super(TankNet, self).__init__()

		feature_width = 128
		mask_width = 5
		post_conv_width = (feature_width - mask_width + 1) / 2
		post_conv_width = (post_conv_width - mask_width + 1) / 2

		self.conv1 = nn.Conv2d(1, 6, mask_width)
		self.conv2 = nn.Conv2d(6, 16, mask_width)
		self.fc1 = nn.Linear(16 * post_conv_width * post_conv_width, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		
		# Convolution and pooling layers
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)

		x = torch.flatten(x, 1)	# This flattens all dimensions except the first (batch dimension)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x)


def train_loop(dataloader: DataLoader, model, loss_fn, optimizer, device):
	
	size = len(dataloader.dataset)
	for batch, (x, y) in enumerate(dataloader):

		x, y = x.to(device), y.to(device)

		# Compute model's prediction and loss
		pred = model(x)
		loss = loss_fn(pred, y)

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 5 == 0:

			loss, current = loss.item(), batch * len(x)
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
			correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

	test_loss /= num_batches
	correct /= size
	print(f'Test results:\n  Accuracy: {(100*correct):.2f}%, Avg loss: {test_loss:.4f}\n')


target_transform = Lambda(
	lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)
)

label_map = {
	'2s1': 0,
	'bmp2': 1,
	'btr70': 2,
	'm1': 3,
	'm2': 4,
	'm35': 5,
	'm60': 6,
	'm548': 7,
	't72': 8,
	'zsu23': 9,
}

images_root_dir = 'qpm\\data'

# Load data
print('Loading data... ', end='')
data_load_start = time.time()

training_data = TankPics(
	img_dir=images_root_dir,
	label_map=label_map,
	train=True,
	transform=ToTensor(),
	target_transform=target_transform,
)

test_data = TankPics(
	img_dir=images_root_dir,
	label_map=label_map,
	train=False,
	transform=ToTensor(),
	target_transform=target_transform,
)

data_load_finish = time.time()
print('%.3f s' % (data_load_finish - data_load_start))

# Hyperparameters
batch_size = 64
learning_rate = 2e-3
epochs = 25
loss_fn = nn.CrossEntropyLoss()

# Dataloaders
train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

# Device settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Need to send model to device before initializing optimizer
model = TankNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):

	print(f'Epoch {epoch + 1}\n-------------------------------')
	train_loop(train_dataloader, model, loss_fn, optimizer, device)
	test_loop(test_dataloader, model, loss_fn, device)

# tank_feature, tank_label = tanks[0]

# plt.imshow(tank_feature.squeeze(), cmap='gray')
# plt.show()
# print(f'Label: {tank_label}')

print('\n========== DONE ==========\n')
