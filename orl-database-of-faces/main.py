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
	return np.array(raster)

if __name__ == '__main__':
	a = read_pgm(open('s1/1.pgm', 'rb'))
	np.save('a.out', a)

