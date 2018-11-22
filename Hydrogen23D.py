import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import copy

import scipy.constants as const
a_0 = const.physical_constants["Bohr radius"][0]
me  = const.m_e
mp  = const.m_p
mu  = me*mp/(me+mp)
pi  = const.pi
hb  = const.hbar
qe  = const.e
e_0  = const.epsilon_0
ev  = const.electron_volt

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.NN=nn.Sequential(
				torch.nn.Linear(6, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 1)
				)
		self.Lambda=nn.Parameter(torch.Tensor([0]))#eigenvalue
		self.alpha=nn.Parameter(torch.Tensor([1]))#coefficient for decay
		self.beta=nn.Parameter(torch.Tensor([1]))#coefficient for decay

	def forward(self,x):
		return self.NN(x)[:,0]*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))


LR=3e-3
BATCH_SIZE=256
H=0.1

net = Net()
params = [p for p in net.parameters()]
opt = torch.optim.Adam(params, lr=LR)

steps=5000
for epoch in range(5):
	start = time.time()
	for step in range(steps):
		print("Progress {:2.0%}".format(step /steps), end="\r")
		X = (torch.rand(BATCH_SIZE,6,requires_grad=True)-0.5)*2
		X = X/torch.norm(X,dim=1)[:,None]*(torch.rand(BATCH_SIZE)[:,None]*5)
		#X = ((torch.rand(BATCH_SIZE,3,requires_grad=True))*5+0.01)
		eps_0 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)
		eps_1 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)

		Psi_0_p = net(X+eps_0)
		Psi_0_n = net(X-eps_0)
		Psi_0   = (Psi_0_p + Psi_0_n)/2


		Psi_1_p = net(X+eps_1)
		Psi_1_n = net(X-eps_1)
		Psi_1   = (Psi_1_p + Psi_1_n)/2

		Lap_0 = a_0**2/2*eps_0*(grad(Psi_0_p,X,create_graph=True,grad_outputs=torch.ones_like(Psi_0_p))[0]-grad(Psi_0_n,X,create_graph=True,grad_outputs=torch.ones_like(Psi_1_p))[0])/(4*H**2)
		Lap_1 = a_0**2/2*eps_1*(grad(Psi_1_p,X,create_graph=True,grad_outputs=torch.ones_like(Psi_0_n))[0]-grad(Psi_1_n,X,create_graph=True,grad_outputs=torch.ones_like(Psi_1_n))[0])/(4*H**2)

		Lap_0 = torch.sum(Lap_0,dim=1)
		Lap_1 = torch.sum(Lap_1,dim=1)
		#r     = torch.norm(X,dim=1)
		R1     = torch.tensor([2,0,0]).type(torch.FloatTensor)
		R2     = torch.tensor([-2,0,0]).type(torch.FloatTensor)
		r11    = torch.norm(X[:,:3]-R1,dim=1)
		r12    = torch.norm(X[:,:3]-R2,dim=1)
		r21    = torch.norm(X[:,3:]-R1,dim=1)
		r22    = torch.norm(X[:,3:]-R2,dim=1)
		rinter = torch.norm(X[:,:3]-X[:,3:],dim=1)


		V     = - (1/r11+1/r12+1/r21+1/r22) + (1/rinter)

		J     = torch.mean((-Lap_0 + (V-net.Lambda)*Psi_0)*(-Lap_1+ (V-net.Lambda)*Psi_1)/(Psi_0*Psi_1))

		opt.zero_grad()
		J.backward()
		opt.step()


	print('e # '+str(epoch+1)+'_____________________________________')
	print('It took', time.time()-start, 'seconds.')
	print('Lambda = '+str(net.Lambda[0].item()))
	print('Alpha  = '+str(net.alpha[0].item()))
	print('Beta   = '+str(net.beta[0].item()))
	#print("E = ",(net.Lambda[0].item())*mu/hb**2/ev)

	X_plot = (torch.rand(5000,6,requires_grad=True)-0.5)*2
	X_plot = X_plot/torch.norm(X_plot,dim=1)[:,None]*(torch.rand(5000)[:,None]*5)
	X_plot[:,1] = 0
	X_plot[:,2] = 0

	Psi_plot = net(X_plot).detach().numpy()
	R_plot = np.linalg.norm(X_plot[:,:3].detach().numpy(),axis=1)*((X_plot[:,3].detach().numpy()>0).astype(int) - (X_plot[:,3].detach().numpy()<=0).astype(int))
	plt.plot(R_plot,(Psi_plot)/max(np.abs(Psi_plot)),marker='.',ls='',ms=1.5,label=(str(epoch)))#+"  energy = "+str(energy)))
plt.legend(loc="upper right")
plt.show()
