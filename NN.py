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

	def forward(self, x):
	
		for i, layer in enumerate(self.layers):
			x = F.elu(layer(x))

			#if not i == (self.depth)-1:
			#	r = F.elu(layer(r))
			#else:
			#	r = torch.sigmoid(layer(r))
		return r
		

class WaveNet(nn.Module):

	def __init__(self,arc,eps=0.1):
		super(WaveNet, self).__init__()
		
		self.eps    = eps
		self.depth  = len(arc)-1
		self.layers = nn.ModuleList([nn.Linear(arc[i],arc[i+1])\
         for i in range(self.depth)])

	def forward(self, x, R):
	
		d = torch.norm((x[:,None,:]-R.repeat(1,x.shape[-1]//R.shape[-1])[None,:,:]).contiguous().view(x.shape[0],-1,R.shape[-1]),dim=-1)
		
		if x.shape[-1]//R.shape[-1] != 1:
			x = x.view(x.shape[0],-1,R.shape[-1])
			de = torch.norm(x[:,:,None,:]-x[:,None,:,:],dim=-1)
			for i in range(x.shape[-2]):
				for j in range(i):
					d = torch.cat((d,de[:,i,j].view(-1,1)),dim=1)
					
		r = torch.erf(d/self.eps)/d
		
		for i, layer in enumerate(self.layers):
			r = F.elu(layer(r))
			
		return r
		
		

