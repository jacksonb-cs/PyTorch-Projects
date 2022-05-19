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

from TankVision import TankNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = TankNet().to(device)
model.load_state_dict(torch.load('TankVision_weights.pth'))