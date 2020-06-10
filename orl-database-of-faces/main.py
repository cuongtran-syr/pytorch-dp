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
	t = torch.empty(size=(400, 1, 112, 92), dtype=torch.uint8)
	i = 0
	for root, dirs, files in os.walk(dirname(__file__), topdown=True):
		dirs.sort(key=lambda w: (len(w), w))
		files.sort(key=lambda w: (len(w), w))
		for filename in files:
			if filename.endswith(f'{os.extsep}pgm'):
				t[i] = read_pgm(open(join(root, filename), 'rb')).unsqueeze(0)
				i += 1
	return t

def train():
	n = Net()

if __name__ == '__main__':
	print(get_data().shape)

