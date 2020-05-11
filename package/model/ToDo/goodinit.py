from model.model import Model, Conv2d_SAME

from copy import deepcopy
from collections import OrderedDict

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GoodInit(Model):
	"""The architecture of GoodInit"""
	def __init__(self, num_classes, data_var):
		super(GoodInit, self).__init__()
		self.num_classes = num_classes # number of classes in classification task
		self.data_var = data_var

		# input_size = [3, 32, 32]
		self.conv1 = Conv2d_SAME(in_channels=3, out_channels=32, kernel_size=3)
		self.conv2 = Conv2d_SAME(in_channels=32, out_channels=32, kernel_size=3)
		self.conv3 = Conv2d_SAME(in_channels=32, out_channels=32, kernel_size=3)
		self.conv4 = Conv2d_SAME(in_channels=32, out_channels=48, kernel_size=3)
		self.conv5 = Conv2d_SAME(in_channels=48, out_channels=48, kernel_size=3)
		self.pool5 = nn.MaxPool2d(kernel_size=2)

		self.conv6 = Conv2d_SAME(in_channels=48, out_channels=80, kernel_size=3)
		self.conv7 = Conv2d_SAME(in_channels=80, out_channels=80, kernel_size=3)
		self.conv8 = Conv2d_SAME(in_channels=80, out_channels=80, kernel_size=3)
		self.conv9 = Conv2d_SAME(in_channels=80, out_channels=80, kernel_size=3)
		self.conv10 = Conv2d_SAME(in_channels=80, out_channels=80, kernel_size=3)
		self.pool10 = nn.MaxPool2d(kernel_size=2)

		self.conv11 = Conv2d_SAME(in_channels=80, out_channels=128, kernel_size=3)
		self.conv12 = Conv2d_SAME(in_channels=128, out_channels=128, kernel_size=3)
		self.conv13 = Conv2d_SAME(in_channels=128, out_channels=128, kernel_size=3)
		self.conv14 = Conv2d_SAME(in_channels=128, out_channels=128, kernel_size=3)
		self.conv15 = Conv2d_SAME(in_channels=128, out_channels=128, kernel_size=3)

		self.fc1 = nn.Linear(in_features=128*1*1, out_features=500)
		self.fc2 = nn.Linear(in_features=500, out_features=self.num_classes)

		if self.data_var == 'cifarcomp':
			self.conv15 = Conv2d_SAME(in_channels=128, out_channels=25, kernel_size=3)
			self.fc1 = nn.Linear(in_features=25*1*1, out_features=500)

		self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.pool5,
					self.conv6, self.conv7, self.conv8, self.conv9, self.conv10, self.pool10,
					self.conv11, self.conv12, self.conv13, self.conv14, self.conv15,
					self.fc1, self.fc2]
		self.trainable_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
								self.conv6, self.conv7, self.conv8, self.conv9, self.conv10,
								self.conv11, self.conv12, self.conv13, self.conv14, self.conv15,
								self.fc1, self.fc2]
		self.layer_trim_objects = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
								self.conv6, self.conv7, self.conv8, self.conv9, self.conv10,
								self.conv11, self.conv12, self.conv13, self.conv14, self.conv15,
								self.fc1]

		self.layer_trim_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',
								'conv6', 'conv7', 'conv8', 'conv9', 'conv10',
								'conv11', 'conv12', 'conv13', 'conv14', 'conv15',
								'fc1']

	def forward(self, x):

		return self.get_prediction(x)

	def get_final_fm(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = self.pool5(x)
		x = F.dropout(x, p=0.25, training=self.training)

		x = F.relu(self.conv6(x))
		x = F.relu(self.conv7(x))
		x = F.relu(self.conv8(x))
		x = F.relu(self.conv9(x))
		x = F.relu(self.conv10(x))
		x = self.pool10(x)
		x = F.dropout(x, p=0.25, training=self.training)

		x = F.relu(self.conv11(x))
		x = F.relu(self.conv12(x))
		x = F.relu(self.conv13(x))
		x = F.relu(self.conv14(x))
		x = F.relu(self.conv15(x))
		# global max pooling
		x = F.max_pool2d(x, kernel_size=x.size()[2:])
		x = F.dropout(x, p=0.25, training=self.training)
		x = x.view(x.size(0), -1) # flatten
		x = F.relu(self.fc1(x))

		x = F.dropout(x, p=0.25, training=self.training)
		return x

	def get_inter_results(self, x):

		inter_zeros = []
		self.layer_neuron_size = []

		x = F.relu(self.conv1(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv2(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv3(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv4(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv5(x))
		inter_zeros.append(self.get_zero_num(x))
		x = self.pool5(x)
		x = F.dropout(x, p=0.25, training=self.training)

		x = F.relu(self.conv6(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv7(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv8(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv9(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv10(x))
		inter_zeros.append(self.get_zero_num(x))
		x = self.pool10(x)
		x = F.dropout(x, p=0.25, training=self.training)

		x = F.relu(self.conv11(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv12(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv13(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv14(x))
		inter_zeros.append(self.get_zero_num(x))
		x = F.relu(self.conv15(x))
		inter_zeros.append(self.get_zero_num(x))
		# global max pooling
		x = F.max_pool2d(x, kernel_size=x.size()[2:])
		x = F.dropout(x, p=0.25, training=self.training)

		x = x.view(x.size(0), -1) # flatten
		x = F.relu(self.fc1(x))
		inter_zeros.append(self.get_zero_num(x)) 
		x = F.dropout(x, p=0.25, training=self.training)
		
		return inter_zeros

	def get_logits(self, x):
		x = self.get_final_fm(x)
		x = self.fc2(x)
		return x

	def get_prediction(self, x):
		x = self.get_logits(x)
		if not self.training:
			x = F.softmax(x, dim=-1)
		return x

	def reconstruct(self, num_classes, classifier_id=0, trainable=True, fe_lst=[0,0,0]):

		self.num_classes = num_classes
		for param in self.parameters():
			param.requires_grad = trainable
		if classifier_id == 0:
			# 1 fc
			self.fc8 = nn.Linear(128, self.num_classes)
		elif classifier_id == 1:
			# 2 fc
			self.fc8 = nn.Sequential(OrderedDict([
							('fc9', nn.Linear(128, 64)),
							('relu9', nn.ReLU()),
							('fc8', nn.Linear(64, self.num_classes))
							]))
		elif classifier_id == 2:
			# conv + fc
			pass

	def define_optimizer(self, train_opt):

		lr = 0.0001
		optimizer = optim.Adam(self.parameters(), lr)
		return optimizer

	def define_criterion(self):
		criterion = nn.CrossEntropyLoss()
		return criterion

	def load_pretrained_weights(self, weights_loc='/home/yujie/model/cifar10/init.h5'):
		#if self.data_var in ['cifar10', 'cifarup']:
		if self.data_var in 'cifar10':
			weights_loc='/home/yujie/model/cifar10/init.h5'
			pretrained_weights = h5py.File(weights_loc, 'r')['model_weights']
			for i, (name, param) in enumerate(self.state_dict().items()):
				layer_name, param_type = name.split('.')
				layer_idx = i // 2
				if layer_name.startswith('conv'):
					match_name = 'conv2d_' + layer_name[4:]
				elif layer_name.startswith('fc'):
					match_name = 'dense_' + layer_name[2:]
		
				if param_type == 'weight':
					pre_weight = pretrained_weights[match_name][match_name]['kernel:0'][:]
					# make the dimension consistent
					if len(pre_weight.shape) == 4:
						# convolutional layer
						pre_weight = pre_weight.transpose(3,2,0,1)
					elif len(pre_weight.shape) == 2:
						# fully-connected layer
						pre_weight = pre_weight.transpose(1,0)

					self.trainable_layers[layer_idx].weight.data = \
														torch.from_numpy(pre_weight)
				elif param_type == 'bias':
					pre_weight = pretrained_weights[match_name][match_name]['bias:0'][:]
					self.trainable_layers[layer_idx].bias.data = \
														torch.from_numpy(pre_weight)

		elif self.data_var == 'cifarcomp':

			weights_loc = '/home/yujie/model/cifarcomp/goodinit_comp.pth'

			self.load_state_dict(torch.load(weights_loc))
		
		elif self.data_var == 'cifarup':

			weights_loc = '/home/yujie/model/cifarup/goodinit.pth'

			self.load_state_dict(torch.load(weights_loc))

		print("********Pretrained Model Loaded from %s!********" % weights_loc)


	def model_reconstruct(self, layer_idx, neuron_to_drop, neuron_to_keep):
		print(layer_idx)
		for i, idx in enumerate(layer_idx):
			print(self.layer_trim_objects[idx])
			if i == 0:
				self.layer_trim_objects[idx].out_features = len(neuron_to_keep[i])
			else:
				assert (layer_idx[i] - layer_idx[i-1]) == 1
				self.layer_trim_objects[idx].out_features = len(neuron_to_keep[i])
				self.layer_trim_objects[idx].in_features = len(neuron_to_keep[i-1])

		self.layer_trim_objects[idx+1].in_features = len(neuron_to_keep[i])

		return

	def modify_model_weights(self, layer_idx, keep_list, drop_list, weights_loc='/home/yujie/model/cifar10/init.h5'):

		pointer = 0
		start = -100
		last = -100
		# add the following layer into the trimming list
		layer_idx.append(layer_idx[-1]+1)
		trim_layer_names = np.array(self.layer_trim_names)[layer_idx]

		pretrained_weights = h5py.File(weights_loc, 'r')['model_weights']
		for i, (name, param) in enumerate(self.state_dict().items()):

			layer_name, param_type = name.split('.')

			if layer_name not in trim_layer_names:

				continue

			layer_idx = i // 2
			if layer_name.startswith('conv'):
				match_name = 'conv2d_' + layer_name[4:]
			elif layer_name.startswith('fc'):
				match_name = 'dense_' + layer_name[2:]
	
			if param_type == 'weight':
				pre_weight = pretrained_weights[match_name][match_name]['kernel:0'][:]
				# make the dimension consistent
				if len(pre_weight.shape) == 4:
					# convolutional layer
					pre_weight = pre_weight.transpose(3,2,0,1)
					if pointer == 0:
						pre_weight = pre_weight[keep_list[pointer],...]
					elif pointer == len(keep_list):
						pre_weight = pre_weight[:, keep_list[pointer-1], ...]
					else:

						print(pre_weight.shape)
						print(pre_weight[keep_list[pointer], keep_list[pointer-1], ...].shape)
						pre_weight = pre_weight[keep_list[pointer], keep_list[pointer-1], ...]

				elif len(pre_weight.shape) == 2:
					# fully-connected layer
					pre_weight = pre_weight.transpose(1,0)

					if pointer == 0:
						pre_weight = pre_weight[keep_list[pointer],...]
					elif pointer == len(keep_list):
						pre_weight = pre_weight[:, keep_list[pointer-1]]
					else:
						pre_weight = pre_weight[keep_list[pointer], keep_list[pointer-1]]

				self.trainable_layers[layer_idx].weight.data = \
													torch.from_numpy(pre_weight)
			elif param_type == 'bias':
				pre_weight = pretrained_weights[match_name][match_name]['bias:0'][:]
				if pointer != len(keep_list):
					pre_weight = pre_weight[keep_list[pointer]]

				self.trainable_layers[layer_idx].bias.data = \
													torch.from_numpy(pre_weight)

			if (i % 2) != 0:
				pointer += 1

		# self.load_state_dict(torch.load(weights_loc))

		print("********Compressed Model Reloaded!********")

		return


