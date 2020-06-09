from os.path import join, dirname

import torch.nn as nn
import numpy as np

class Net(nn.Module):

	def __init__():
		pass
	
	def forward():
		pass

def read_pgm(pgmf):
	"""Return a raster of integers from a PGM as a list of lists."""
	a = pgmf.readline().decode('utf-8')
	assert a == 'P5\n', a
	(width, height) = [int(i) for i in pgmf.readline().split()]
	depth = int(pgmf.readline())
	assert depth <= 255

	#return np.fromfile(pgmf, dtype=np.int16)
	raster = []
	for y in range(height):
		row = []
		for y in range(width):
			row.append(ord(pgmf.read(1)))
		raster.append(row)
	return np.array(raster, np.uint8)

def display_example():
	from matplotlib import pyplot as plt
	plt.imshow(read_pgm(open(join(dirname(__file__), 's1', '1.pgm'), 'rb')), plt.cm.gray)
	plt.show()

if __name__ == '__main__':
	display_example()

