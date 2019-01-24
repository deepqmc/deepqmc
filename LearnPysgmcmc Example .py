from itertools import islice
import sys
sys.path.insert(0, "../../../")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import copy
import datetime
import scipy as sp
cmap=plt.get_cmap("plasma")







class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.NN=nn.Sequential(
					torch.nn.Linear(1, 64),
					torch.nn.ELU(),
					torch.nn.Linear(64, 64),
					torch.nn.ELU(),
					#torch.nn.Linear(64, 64),
					#torch.nn.ELU(),
					torch.nn.Linear(64, 64),
					torch.nn.ELU(),
					torch.nn.Linear(64, 1)
					)
			self.Lambda=nn.Parameter(torch.Tensor([-1]))	#eigenvalue


		def forward(self,x):
			d = ((x[0]**2+x[1]**2)**(1/2)).view(-1,1)              #define forward pass
			r = torch.erf(d/0.1)/d         #get inverse distances
			return self.NN(r)[:,0]

net=Net()

def test(x):
	return -torch.log(net(x)**2)
	
def banana_nll(x):
	ll = -(0.5*(0.01*x[0]**2+(x[1] + 0.1 * x[0]**2 -10)**2))
	return -ll


def plot_banana(grid=(np.arange(-25, 25, 0.05),np.arange(-25, 25, 0.05)),):
	x, y = grid
	xx, yy = np.meshgrid(x, y, sparse=True)
	densities = np.asarray([np.exp(-banana_nll((x, y))) for x in xx for y in yy])
	plt.figure()
	plt.contour(x, y, densities, 1)
	plt.xlim(xmin=-30, xmax=30)
	plt.ylim(ymin=-40, ymax=40)
	plt.show()

#plot_banana()
	

def plot_samples(sampler_cls=SGHMC, lr=0.1, num_burn_in_steps=3000,num_samples=1000,color="b"):

	rand1 = torch.rand(1, requires_grad=True)
	rand2 = torch.rand(1, requires_grad=True)
	#print(help(sampler_cls))
	sampler = sampler_cls(params=(tmp,tmp2),lr=lr, negative_log_likelihood=test)
	
	# skip burn-in samples
	_ = [sample for sample, _ in islice(sampler, num_burn_in_steps)]
	samples = np.asarray([sample for sample, _ in islice(sampler, num_samples)])
	plt.scatter(samples[:, 0], samples[:, 1], color=color, label=sampler_cls.__name__)
	plt.legend()
	plt.show()
	

plot_samples(SGHMC, lr=0.5)

	
	
