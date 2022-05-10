import os
from torchvision.io import read_image

i = 0

for root, dirs, files in os.walk('qpm\\test_data'):

	i += 1
	print(f'Iter: {i}')
	print(f'Root: {root}\nDirs: {dirs}\nFiles: {files}\n')

	# for name in files:

	# 	print(os.path.join(root, name))

	# for name in dirs:

	# 	print(os.path.join(root, name))

