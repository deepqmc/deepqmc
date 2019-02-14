import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):

	def __init__(self,arc):
		super(Net, self).__init__()

		self.depth = len(arc)-1
		self.layers = nn.ModuleList([nn.Linear(arc[i],arc[i+1])\
         for i in range(self.depth)])

	def forward(self, x, R):

		d = torch.norm((x-R[:,None,:]),dim=2).transpose(0,1)
		r = torch.erf(d/0.1)/d
		for i, layer in enumerate(self.layers):
			r = F.elu(layer(r))

			#if not i == (self.depth)-1:
			#	r = F.elu(layer(r))
			#else:
			#	r = torch.sigmoid(layer(r))
		return r
