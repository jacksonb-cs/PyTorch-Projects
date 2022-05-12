from ast import Lambda
import os

import matplotlib.pyplot as plt

from skimage import io

import torch
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
		# This is really ugly...
		for file in reversed(self.file_names):
			
			suffix = file.split('Deg_0', 1)[1]	# Get second half of split
			degree = int(suffix.split('_', 1)[0])	# Get first half of split as integer

			if self.train and degree > 16:

				self.file_names.remove(file)

			elif not self.train and degree < 17:

				self.file_names.remove(file)


	def __len__(self):
		
		return len(self.file_names)


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

images_dir = 'qpm\\data'
batch_size = 64

training_data = TankPics(
	img_dir=images_dir,
	label_map=label_map,
	train=True,
	transform=ToTensor(),
	target_transform=target_transform,
)

test_data = TankPics(
	img_dir=images_dir,
	label_map=label_map,
	train=False,
	transform=ToTensor(),
	target_transform=target_transform,
)

train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

# tank_feature, tank_label = tanks[0]

# plt.imshow(tank_feature.squeeze(), cmap='gray')
# plt.show()
# print(f'Label: {tank_label}')

print('\n========== DONE ==========\n')
