import os
from os.path import join, dirname

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

class Net(nn.Module):

	def __init__():
		super().__init__()
	
	def forward(t):
		return t

def read_pgm(pgmf):
	"""Return a raster of integers from a PGM as a list of lists."""
	#https://stackoverflow.com/a/35726744/7102572
	assert pgmf.readline().decode('utf-8') == 'P5\n'
	(width, height) = [int(i) for i in pgmf.readline().split()]
	depth = int(pgmf.readline())
	assert depth <= 255

	raster = []
	for y in range(height):
		row = []
		for y in range(width):
			row.append(ord(pgmf.read(1)))
		raster.append(row)
	return torch.tensor(raster, dtype=torch.uint8)

def display_example():
	plt.imshow(read_pgm(open(join(dirname(__file__), 's1', '1.pgm'), 'rb')), plt.cm.gray)
	plt.show()

def get_data():
	train = torch.empty(size=(280, 1, 112, 92), dtype=torch.uint8)
	test = torch.empty(size=(120, 1, 112, 92), dtype=torch.uint8)
	i = 0
	for root, dirs, files in os.walk(dirname(__file__), topdown=True):
		dirs.sort(key=lambda w: (len(w), w)) #sort by length first, then lexicographically
		files.sort(key=lambda w: (len(w), w)) #sort by length first, then lexicographically
		for filename in files:
			if filename.endswith(f'{os.extsep}pgm'):
				if i % 10 < 7:
					train[(i % 10) + (i // 10) * 7] = read_pgm(open(join(root, filename), 'rb')).unsqueeze(0)
				else:
					test[((i - 7) % 10) + (i // 10) * 3] = read_pgm(open(join(root, filename), 'rb')).unsqueeze(0)
				i += 1
	return train, test

def write_data():
	train, test = get_data()
	train_path = join(dirname(__file__), f'train{os.extsep}pt')
	test_path = join(dirname(__file__), f'test{os.extsep}pt')
	torch.save(train, train_path)
	torch.save(test, test_path)
	print(f'Successfully saved "{train_path}" and "{test_path}"')

def train():
	n = Net()

if __name__ == '__main__':
	write_data()

