import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import copy



def metropolis(distribution,interval,startpoint,maxstepsize,steps,presteps=0):
	"""Metropolis sampling from distribution.

		Parameter:
			distribution = distribution to be sampled of (network,torch)
			interval     = interval to be sampled from
			startpoint   = start of the random walker
			maxstepsize  = maximum size of random steps
			steps        = number of steps to be proposed
			presteps     = number of steps proposed before sampling

		Returns:
			array containing samples from the distribution"""
	#get dimension
	dim = startpoint.shape
	#convert interval to torch Tensor
	interval = torch.tensor(interval).type(torch.FloatTensor)
	#initialise list to store walker positions
	samples = []
	#initialise list to store respective propability
	prob = []
	#initialise the walker at the startposition
	walker = torch.tensor([startpoint]).type(torch.FloatTensor)
	walker_prob = distribution(walker)
	#loop over proposal steps
	for i in range(presteps+steps):
		#append position of walker to the sample list in case presteps exceeded
		if i > (presteps-1):
			samples.append(walker)
			prob.append(walker_prob)

		#propose new trial position
		trial = walker + torch.from_numpy(np.random.uniform(-maxstepsize,maxstepsize,size=dim)).type(torch.FloatTensor)
		#check if in interval
		inint = torch.tensor((interval[0]<trial).all() and (trial<interval[1]).all()).type(torch.FloatTensor)
		#calculate trial propability
		trial_prob = distribution(trial)*inint
		#calculate acceptance propability
		ratio = trial_prob/walker_prob
		#accept trial position with respective propability
		if ratio > np.random.uniform(0,1):
			walker=trial
			walker_prob=trial_prob
	#return list of samples
	return torch.stack(samples).view(-1,dim[0]),torch.stack(prob).view(-1)


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.NN=nn.Sequential(
				torch.nn.Linear(2, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 1)
				)
		self.Lambda=nn.Parameter(torch.Tensor([-1]))	#eigenvalue
		#self.alpha=nn.Parameter(torch.Tensor([1]))#coefficient for decay
		#self.beta=nn.Parameter(torch.Tensor([8]))#coefficient for decay
		self.alpha=nn.Parameter(torch.Tensor([1]))#coefficient for decay
		self.beta=nn.Parameter(torch.Tensor([5]))#coefficient for decay
	def forward(self,x):
		d = torch.zeros(len(x),2)
		d[:,0] = torch.norm(x-R1,dim=1)
		d[:,1] = torch.norm(x-R2,dim=1)
		return self.NN(d)[:,0]*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))

def analytical_gs(r,a_0=1):
	return 2*(a_0)**(-3/2)*np.exp(-r/(a_0))


LR=1e-3
BATCH_SIZE=2056
steps=1000
epochs=10


H=0.2 #smoothing


R1    = torch.tensor([1.5,0,0]).type(torch.FloatTensor)
R2    = torch.tensor([-1.5,0,0]).type(torch.FloatTensor)
R     = torch.norm(R1-R2)


for test in range(1):

	net = Net()
	params = [p for p in net.parameters()]
	#del params[0]
	del params[1]
	del params[1]

	opt = torch.optim.Adam(params, lr=LR)
	for epoch in range(epochs):
		start = time.time()
		# with torch.no_grad():
		# 	X_all,Psi_t_all = metropolis(lambda x :net(x)**2,(-6*np.array([1,1,1]),6*np.array([1,1,1])),np.array([0,0,0]),2,BATCH_SIZE*steps,presteps=10)
		# 	indx = torch.randperm(steps*BATCH_SIZE)
		#X_all.requires_grad = True
		for step in range(steps):
			print("Progress {:2.0%}".format(step /steps), end="\r")
			# X = X_all[indx[BATCH_SIZE*step:(BATCH_SIZE*(step+1))]]
			# Psi_t = Psi_t_all[indx[BATCH_SIZE*step:BATCH_SIZE*(step+1)]]
			X = (torch.rand(BATCH_SIZE,3,requires_grad=True)-0.5)*2*5
			#X = torch.from_numpy(np.random.normal(0,0.2,(BATCH_SIZE,3))).type(torch.FloatTensor)
			#X.requires_grad = True
			eps_0 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)
			eps_1 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)

			Psi_0_p = net(X+eps_0)
			Psi_0_n = net(X-eps_0)
			Psi_0   = (Psi_0_p + Psi_0_n)/2

			Psi_1_p = net(X+eps_1)
			Psi_1_n = net(X-eps_1)
			Psi_1   = (Psi_1_p + Psi_1_n)/2

			Lap_0 = eps_0*(grad(Psi_0_p,X,create_graph=True,grad_outputs=torch.ones_like(Psi_0_p))[0]\
			-grad(Psi_0_n,X,create_graph=True,grad_outputs=torch.ones_like(Psi_1_p))[0])/(4*H**2)
			Lap_1 = eps_1*(grad(Psi_1_p,X,create_graph=True,grad_outputs=torch.ones_like(Psi_0_n))[0]\
			-grad(Psi_1_n,X,create_graph=True,grad_outputs=torch.ones_like(Psi_1_n))[0])/(4*H**2)
			#
			Lap_0 = torch.sum(Lap_0,dim=1)
			Lap_1 = torch.sum(Lap_1,dim=1)
			r1    = torch.norm(X-R1,dim=1)
			r2    = torch.norm(X-R2,dim=1)
			V     = -1/r1 - 1/r2 + 1/R
			# w     = Psi_0*Psi_1/Psi_t**2
			# J     = torch.sum(w*(-Lap_0/Psi_0 + (V-net.Lambda))*(-Lap_1/Psi_1+ (V-net.Lambda)))/torch.sum(Psi_0*Psi_1/Psi_t)
			G=torch.meshgrid([torch.linspace(-5,5,10),torch.linspace(-5,5,10),torch.linspace(-5,5,10)])
			x=G[0].flatten().view(-1,1)
			y=G[1].flatten().view(-1,1)
			z=G[2].flatten().view(-1,1)
			S = torch.cat((x, y, z), 1)
			S.requires_grad=True
			symloss=torch.sum((net(S[:10**3//2])-net(S[10**3//2:]))**2)/torch.sum(net(S)**2)

			J     = torch.sum(((-Lap_0/Psi_0 + V-net.Lambda)*(-Lap_1/Psi_1+ V-net.Lambda))) + 15*(epoch/epochs)*symloss
			opt.zero_grad()
			J.backward()
			opt.step()

		G=torch.meshgrid([torch.linspace(-5,5,50),torch.linspace(-5,5,50),torch.linspace(-5,5,50)])
		x=G[0].flatten().view(-1,1)
		y=G[1].flatten().view(-1,1)
		z=G[2].flatten().view(-1,1)
		X = torch.cat((x, y, z), 1)
		X.requires_grad=True
		gPsi  = grad(net(X),X,create_graph=True,grad_outputs=torch.ones(len(X)))[0]
		r1    = torch.norm(X-R1,dim=1)
		r2    = torch.norm(X-R2,dim=1)
		V     = -1/r1 - 1/r2 + 1/R
		Psi   = net(X)
		E     = torch.mean(torch.sum(gPsi**2,dim=1)/2+Psi**2*V)/torch.mean(Psi**2) #energy still is somewhat strange, cause note very stable even if functions look similar


		print('e # '+str(epoch+1)+'_____________________________________')
		print('It took', time.time()-start, 'seconds.')
		print('Lambda = '+str(net.Lambda[0].item()*27.2))
		print('Energy = '+str(E.item()*27.2))
		print('Alpha  = '+str(net.alpha[0].item()))
		print('Beta   = '+str(net.beta[0].item()))


		X_plot = torch.from_numpy(np.swapaxes(np.array([np.linspace(-6,6,100),np.zeros(100),np.zeros(100)]).reshape(3,100),0,1)).type(torch.FloatTensor)
		Psi_plot = net(X_plot).detach().numpy()
		if Psi_plot[np.argmax(np.abs(Psi_plot))] < 0:
			Psi_plot *= -1
		plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(epoch),ls=':')


	X_plot = torch.from_numpy(np.swapaxes(np.array([np.linspace(-6,6,100),np.zeros(100),np.zeros(100)]).reshape(3,100),0,1)).type(torch.FloatTensor)
	Psi_plot = net(X_plot).detach().numpy()
	if Psi_plot[np.argmax(np.abs(Psi_plot))] < 0:
		Psi_plot *= -1
	plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E.item()*27.2,2)))


plt.axvline(R1.numpy()[0],ls=':',color='k')
plt.axvline(R2.numpy()[0],ls=':',color='k')

plt.legend(loc="upper right")
plt.show()
