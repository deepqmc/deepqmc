import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import copy
import datetime

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.NN=nn.Sequential(
				torch.nn.Linear(2, 64),
				torch.nn.ELU(),
				torch.nn.Linear(64, 64),
				torch.nn.ELU(),
				torch.nn.Linear(64, 64),
				torch.nn.ELU(),
				torch.nn.Linear(64, 1)
				)
		self.Lambda=nn.Parameter(torch.Tensor([-1]))	#eigenvalue
		self.alpha=nn.Parameter(torch.Tensor([1]))#coefficient for decay
		self.beta=nn.Parameter(torch.Tensor([8]))#coefficient for decay
	def forward(self,x):

		d = torch.zeros(len(x),2)
		d[:,0] = torch.norm(x-R1,dim=1)
		d[:,1] = torch.norm(x-R2,dim=1)
		return self.NN(d)[:,0]*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))

net = Net()

R1=torch.tensor(1).type(torch.FloatTensor)
R2=torch.tensor(-1).type(torch.FloatTensor)
R=R1-R2

params = [p for p in net.parameters()]
del params[0]
#del params[1]
#del params[1]
opt = torch.optim.Adam(params, lr=1e-3)


for i in range(1000):

	X1_all = torch.from_numpy(np.random.normal(0,1,(100,1))*3/2*R.numpy()).type(torch.FloatTensor)
	X2_all = torch.from_numpy(np.random.normal(0,1,(100,1))*3/2*R.numpy()).type(torch.FloatTensor)
	X3_all = torch.from_numpy(np.random.normal(0,1,(100,1))*3/2*R.numpy()).type(torch.FloatTensor)
	X1_all.requires_grad = True
	X2_all.requires_grad = True
	X3_all.requires_grad = True
	X_all=torch.cat([X1_all,X2_all,X3_all], dim=0).reshape(3,100).transpose(0,1)


	
	Psi=net(X_all).flatten()
	
	dx =torch.autograd.grad(Psi,X1_all,create_graph=True,retain_graph=True,grad_outputs=torch.ones(100))[0]
	ddx=torch.autograd.grad(dx.flatten(),X1_all,retain_graph=True,grad_outputs=torch.ones(100))[0]
	dy =torch.autograd.grad(Psi,X2_all,create_graph=True,retain_graph=True,grad_outputs=torch.ones(100))[0]
	ddy=torch.autograd.grad(dy.flatten(),X2_all,retain_graph=True,grad_outputs=torch.ones(100))[0]
	dz =torch.autograd.grad(Psi,X3_all,create_graph=True,retain_graph=True,grad_outputs=torch.ones(100))[0]
	ddz=torch.autograd.grad(dz.flatten(),X3_all,retain_graph=True,grad_outputs=torch.ones(100))[0]
	lap_X = (ddx+ddy+ddz).flatten()


	loss = torch.sum(-lap_X*Psi)/torch.sum(Psi**2)
	opt.zero_grad()
	loss.backward()
	opt.step()

X1_plot = torch.linspace(-6,6,100)
X2_plot = torch.zeros(100)
X3_plot = torch.zeros(100)
X1_plot.requires_grad = True
X2_plot.requires_grad = True
X3_plot.requires_grad = True

X_plot  =  torch.cat([X1_plot,X2_plot,X3_plot], dim=0).reshape(3,100).transpose(0,1)


Psi = net(X_plot)
dx =torch.autograd.grad(Psi,X1_plot,create_graph=True,retain_graph=True,grad_outputs=torch.ones(100))[0]
ddx=torch.autograd.grad(dx.flatten(),X1_plot,retain_graph=True,grad_outputs=torch.ones(100))[0]
dy =torch.autograd.grad(Psi,X2_plot,create_graph=True,retain_graph=True,grad_outputs=torch.ones(100))[0]
#ddy=torch.autograd.grad(dy[0].flatten(),X2,retain_graph=True,grad_outputs=torch.ones(100))[0]
dz =torch.autograd.grad(Psi,X3_plot,create_graph=True,retain_graph=True,grad_outputs=torch.ones(100))[0]
#ddz=torch.autograd.grad(dz[0].flatten(),X3,retain_graph=True,grad_outputs=torch.ones(100))[0]
#lap_X = (ddx+ddy+ddz).flatten()


plt.plot(X_plot.detach().numpy()[:,0],Psi.detach().numpy(),label="Psi")
plt.plot(X_plot.detach().numpy()[:,0],dx.detach().numpy(),label="dPsi/dx")
plt.plot(X_plot.detach().numpy()[:,0],ddx.detach().numpy(),label="d²Psi/dx²")
plt.legend()
plt.show()
