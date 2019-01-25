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
import seaborn as sb
import pandas as pd
import pymc3 as pm

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
			d = torch.norm(x,dim=1).view(-1,1)              #define forward pass
			r = torch.erf(d/0.1)/d         #get inverse distances
			return self.NN(r)[:,0]

net=Net()
x,y = torch.meshgrid([torch.linspace(-2,2,20),torch.linspace(-2,2,20)])
G=torch.cat((x,y)).view(2,20,20).transpose(0,-1)
P=torch.zeros(20,20)
for i in range(20):

	P[i] = net(G[i])**2

def my_likelihood(x):
	print(x)
	#x = torch.from_numpy(x)
	return net(x)**2

#def my_likelihood(x,tau=0.1):
#	return np.sqrt(tau/(2*np.pi))*np.exp(-tau/2*x**2)
	#return np.exp(-0.1*x**2)
# model = pm.Model()
#
# with model:
#     out = pm.Normal("model", mu=0, tau=0.1, shape=1)
# 	#out = pm.DensityDist("model", my_likelihood, shape=1, testval=0)
#
# with model:
#     step = pm.Metropolis()
#     trace = pm.sample(1000, tune=5, init=None, chains=1, step=step, cores=1)
#
# plt.subplot2grid((2,1),(0,0))
# plt.plot(np.linspace(-10,10,100),my_likelihood(np.linspace(-10,10,100)),color='r')
# plt.hist(trace['model'],bins=50,range=(-10,10),normed=True)
# del model
model = pm.Model()

with model:
    #out = pm.Normal("model", mu=0, tau=0.1, shape=1)
	out = pm.DensityDist("model", my_likelihood, shape=1, testval=1)

with model:
    step = pm.Metropolis()
    trace = pm.sample(1000, tune=5, init=None, chains=1, step=step, cores=1)
plt.subplot2grid((2,1),(1,0))
plt.plot(np.linspace(-10,10,100),my_likelihood(np.linspace(-10,10,100)),color='r')
plt.hist(trace['model'],bins=50,range=(-10,10),normed=True)

plt.show()
