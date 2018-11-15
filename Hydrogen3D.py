import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import copy

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.NN=nn.Sequential(
				torch.nn.Linear(3, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 1)
				)
		self.Lambda=nn.Parameter(torch.Tensor([0]))#eigenvalue
		self.alpha=nn.Parameter(torch.Tensor([1.]))#coefficient for decay
		self.beta=nn.Parameter(torch.Tensor([1.]))#coefficient for decay

	def forward(self,x):
		return self.NN(x)[:,0]*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))


LR=1e-3
BATCH_SIZE=256
H=0.1

net = Net()
params = [p for p in net.parameters()]
opt = torch.optim.Adam(params, lr=LR)

steps=2000
for epoch in range(5):
	start = time.time()	
	for step in range(steps):
		print("Progress {:2.0%}".format(step /steps), end="\r")
		X = (torch.rand(BATCH_SIZE,3,requires_grad=True))*5+0.01
		eps_0 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)
		eps_1 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)	

		Psi_0_p = net(X+eps_0)
		Psi_0_n = net(X-eps_0)
		Psi_0   = (Psi_0_p + Psi_0_n)/2
		

		Psi_1_p = net(X+eps_1)
		Psi_1_n = net(X-eps_1)
		Psi_1   = (Psi_1_p + Psi_1_n)/2

		Lap_0 = eps_0*(grad(Psi_0_p,X,create_graph=True,grad_outputs=torch.ones_like(Psi_0_p))[0]-grad(Psi_0_n,X,create_graph=True,grad_outputs=torch.ones_like(Psi_1_p))[0])/(4*H**2)
		Lap_1 = eps_1*(grad(Psi_1_p,X,create_graph=True,grad_outputs=torch.ones_like(Psi_0_n))[0]-grad(Psi_1_n,X,create_graph=True,grad_outputs=torch.ones_like(Psi_1_n))[0])/(4*H**2)
		
		Lap_0 = torch.sum(Lap_0,dim=1)
		Lap_1 = torch.sum(Lap_1,dim=1)
		r     = torch.norm(X,dim=1)


		J     = torch.mean((-Lap_0 + (-1/r-net.Lambda)*Psi_0)*(-Lap_1+ (-1/r-net.Lambda)*Psi_1)/(Psi_0*Psi_1))

		opt.zero_grad()
		J.backward()
		opt.step()

	print('e # '+str(epoch+1)+'_____________________________________')
	print('It took', time.time()-start, 'seconds.')
	print('Lambda = '+str(net.Lambda[0].item()))
	print('Alpha  = '+str(net.alpha[0].item()))
	print('Beta   = '+str(net.beta[0].item()))

	X_plot = (torch.rand(5000,3))*5+0.01
	Psi_plot = net(X_plot).detach().numpy()
	R_plot = np.linalg.norm(X_plot.detach().numpy(),axis=1) 
	plt.plot(R_plot,np.abs(Psi_plot),marker='.',ls='',label=epoch)
plt.legend(loc="upper right")
plt.show()	




