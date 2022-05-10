import os

import matplotlib.pyplot as plt

from skimage import io

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.io import read_image

class TankPics(Dataset):

	def __init__(self, img_dir, transform=None, target_transform=None):
		
		self.img_labels = []
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

		for _, _, files in os.walk(img_dir):
			
			self.img_labels.extend(files)

	def __len__(self):
		
		return len(self.img_labels)

	def __getitem__(self, index):
		
		file_name = self.img_labels[index]
		label = file_name.split('_', 1)[0]
		img_path = os.path.join(self.img_dir, label, file_name)
		image = io.imread(img_path)

		if self.transform:
			image = self.transform(image)

		if self.target_transform:
			label = self.target_transform(label)

		return image, label

tanks = TankPics('qpm\\data')

boom_car, boom_car_type = tanks[200]

plt.imshow(boom_car, cmap='gray')
plt.show()
print(f'Label: {boom_car_type}')

print(f'Dataset size: {len(tanks)} pictures')

# TO-DO: Write target_transform!

print('\n========== DONE ==========\n')
