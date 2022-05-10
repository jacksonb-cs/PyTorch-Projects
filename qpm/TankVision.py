from ast import Lambda
import os

import matplotlib.pyplot as plt

from skimage import io

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Lambda

class TankPics(Dataset):

	def __init__(self, img_dir, label_map: dict=None, transform=None, target_transform=None):
		
		self.img_dir = img_dir
		self.label_to_int = label_map
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

tanks = TankPics(
	img_dir='qpm\\data',
	label_map=label_map,
	transform=ToTensor(),
	target_transform=Lambda(
		lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)
	),
)

# tank_feature, tank_label = tanks[750]

# plt.imshow(tank_feature.squeeze(), cmap='gray')
# plt.show()
# print(f'Label: {tank_label}')

# TO-DO: Write target_transform!

print('\n========== DONE ==========\n')
