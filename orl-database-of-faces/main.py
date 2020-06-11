import os
from os.path import join, dirname
import random
import sys

import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	from torchdp import PrivacyEngine

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

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
	plt.imshow(read_pgm(open(join(os.curdir, dirname(__file__), 's1', '1.pgm'), 'rb')), plt.cm.gray)
	plt.show()

def get_data():
	train_features = torch.empty(size=(280, 1, 112, 92), dtype=torch.uint8)
	train_labels = torch.empty(size=(280,), dtype=torch.uint8)
	test_features = torch.empty(size=(120, 1, 112, 92), dtype=torch.uint8)
	test_labels = torch.empty(size=(120,), dtype=torch.uint8)
	i = 0
	for root, dirs, files in os.walk(join(os.curdir, dirname(__file__)), topdown=True):
		dirs.sort(key=lambda w: (len(w), w)) #sort by length first, then lexicographically
		files.sort(key=lambda w: (len(w), w)) #sort by length first, then lexicographically
		for filename in files:
			if filename.endswith(f'{os.extsep}pgm'):
				if i % 10 < 7:
					train_features[(i % 10) + (i // 10) * 7] = read_pgm(open(join(root, filename), 'rb')).unsqueeze(0)
					train_labels[(i % 10) + (i // 10) * 7] = (i // 10)
				else:
					test_features[((i - 7) % 10) + (i // 10) * 3] = read_pgm(open(join(root, filename), 'rb')).unsqueeze(0)
					test_labels[((i - 7) % 10) + (i // 10) * 3] = (i // 10)
				i += 1
	return train_features, train_labels, test_features, test_labels

def write_data():
	train_features, train_labels, test_features, test_labels = get_data()
	train_features_path = join(os.curdir, dirname(__file__), f'train_features{os.extsep}pt')
	train_labels_path = join(os.curdir, dirname(__file__), f'train_labels{os.extsep}pt')
	test_features_path = join(os.curdir, dirname(__file__), f'test_features{os.extsep}pt')
	test_labels_path = join(os.curdir, dirname(__file__), f'test_labels{os.extsep}pt')
	torch.save(train_features, train_features_path)
	torch.save(train_labels, train_labels_path)
	torch.save(test_features, test_features_path)
	torch.save(test_labels, test_labels_path)
	print(f'Successfully saved "{train_features_path}", "{train_labels_path}", "{test_features_path}", "{test_labels_path}"')

def train(architecture='softmax'):
	n = nn.Sequential(
		nn.Flatten(),
		nn.Linear(in_features=112 * 92, out_features=40),
	) if architecture == 'softmax' else nn.Sequential(
		nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2, stride=1),
		nn.Flatten(),
		nn.Linear(in_features=112 * 92, out_features=40),
	) if architecture == 'conv 1 channel' else nn.Sequential(
		nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, padding=2, stride=1),
		nn.Flatten(),
		nn.Linear(in_features=112 * 92 * 3, out_features=40),
	) if architecture == 'conv 3 channel' else nn.Sequential(
		nn.Flatten(),
		nn.Linear(in_features=112 * 92, out_features=1500),
		nn.ReLU(),
		nn.Linear(in_features=1500, out_features=40),
	)
	lr = 0.01
	optimizer = torch.optim.Adam(n.parameters(), lr=lr)

	train_features = torch.load(join(os.curdir, dirname(__file__), f'train_features{os.extsep}pt')).float()
	train_labels = torch.load(join(os.curdir, dirname(__file__), f'train_labels{os.extsep}pt')).long()
	test_features = torch.load(join(os.curdir, dirname(__file__), f'test_features{os.extsep}pt')).float()
	test_labels = torch.load(join(os.curdir, dirname(__file__), f'test_labels{os.extsep}pt')).long()

	if len(sys.argv) > 1:
		privacy_engine = PrivacyEngine(
			n,
			batch_size=train_labels.shape[0],
			sample_size=train_labels.shape[0],
			alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
			noise_multiplier=float(sys.argv[1]),
			max_grad_norm=1.5,
		)
		privacy_engine.attach(optimizer)

	train_losses = []
	test_losses = []
	train_accuracy = []
	test_accuracy = []
	print(f'Train Network {architecture} with learning rate {lr}' + (f' and sigma {float(sys.argv[1])}' if len(sys.argv) > 1 else ''))
	num_epochs = 101
	with tqdm(total=num_epochs, dynamic_ncols=True) as pbar:
		for i in range(num_epochs):
			pred_train_labels = n(train_features)
			loss = F.cross_entropy(pred_train_labels, train_labels)
			train_losses.append(loss.item())
			train_accuracy.append((pred_train_labels.max(axis=1).indices == train_labels).sum().item() / len(train_labels))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % 5 == 0:
				n.eval()
				with torch.no_grad():
					pred_test_labels = n(test_features)
					loss = F.cross_entropy(pred_test_labels, test_labels)
					test_losses.append((i, loss.item()))
					test_accuracy.append((i, (pred_test_labels.max(axis=1).indices == test_labels).sum().item() / len(test_labels)))
				n.train()
			if len(sys.argv) > 1:
				delta = 1e-5
				epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
				pbar.set_description(f'Loss = {np.mean(train_losses):.4f}, (ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}')
			pbar.update(1)

	with torch.no_grad():
		n.eval()
		print(f'Train performance: {(n(train_features).max(axis=1).indices == train_labels).sum().item() / len(train_labels) * 100:.2f}%')
		print(f'Test performance: {(n(test_features).max(axis=1).indices == test_labels).sum().item() / len(test_labels) * 100:.2f}%')
		plt.plot(range(len(train_losses)), train_losses, label='Train loss')
		plt.plot([t[0] for t in test_losses], [t[1] for t in test_losses], label='Validation loss')
		plt.legend()
		plt.title('Loss of training and validation')
		plt.show()
		plt.plot(range(len(train_accuracy)), train_accuracy, label='Train accuracy')
		plt.plot([t[0] for t in test_accuracy], [t[1] for t in test_accuracy], label='Validation accuracy')
		plt.legend()
		plt.title('Accuracy of training and validation')
		plt.show()

	model_invert(6, 200, 0.01, n)

def model_invert(label, max_steps, learning_rate, net):
	torch.set_grad_enabled(True)
	net.eval()
	x = torch.autograd.Variable(torch.zeros(size=(1, 1, 112, 92), dtype=torch.float, requires_grad=True), requires_grad=True)
	x_min = x
	c_min = float('inf')
	print('Model inversion')
	for step in trange(max_steps):
		net.zero_grad()
		cost = 1 - net(x)[0, label]
		cost.backward()
		x = torch.autograd.Variable(x - learning_rate * x.grad, requires_grad=True)
		if c_min > cost:
			c_min = cost
			x_min = x
	plt.imshow(x_min.detach().numpy()[0][0], plt.cm.gray)
	plt.show()

if __name__ == '__main__':
	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	train()

