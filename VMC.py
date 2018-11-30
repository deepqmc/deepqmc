import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import copy

cmap=plt.get_cmap("plasma")


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
		self.alpha=nn.Parameter(torch.Tensor([1]))#coefficient for decay
		self.beta=nn.Parameter(torch.Tensor([5]))#coefficient for decay
	def forward(self,x):
		d = torch.zeros(len(x),2)
		d[:,0] = torch.norm(x-R1,dim=1)
		d[:,1] = torch.norm(x-R2,dim=1)
		return self.NN(d)[:,0]*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))



LR=1e-3
BATCH_SIZE=2056
steps=1000
epochs=16


H=0.2 #smoothing


R1    = torch.tensor([1.5,0,0]).type(torch.FloatTensor)
R2    = torch.tensor([-1.5,0,0]).type(torch.FloatTensor)
R     = torch.norm(R1-R2)


gd=6 #number of gridpoints for evaluating symmetry loss
G=torch.meshgrid([torch.linspace(-5,5,gd),torch.linspace(-5,5,gd),torch.linspace(-5,5,gd)])
x=G[0].flatten().view(-1,1)
y=G[1].flatten().view(-1,1)
z=G[2].flatten().view(-1,1)
S = torch.cat((x, y, z), 1)
t = torch.tensor([-1,1,1]).type(torch.FloatTensor)
S_left = S.view(2,gd//2,-1,3)[0].view(-1,3)
S_right = torch.flip(S.view(2,gd//2,-1,3)[1],dims=(0,)).view(-1,3)
S.requires_grad=True
S_left.requires_grad=True
S_right.requires_grad=True


for test in range(1):

	net = Net()
	params = [p for p in net.parameters()]
	del params[0]
	#del params[1]
	#del params[1]
	opt = torch.optim.Adam(params, lr=LR)
	
	for epoch in range(epochs):
	
		print("epoch " +str(1+epoch)+" of "+str(epochs)+":")
		if (epoch)%4 == 1:
			print("symmetrize")
		elif (epoch)%4 == 3:
			print("minimize energy")
		else:
			print("minimize variance of energy")
			
		print("Progress {:2.0%}".format(step /steps), end="\r")
		
		start = time.time()
		
		# with torch.no_grad():
		# 	X_all,Psi_t_all = metropolis(lambda x :net(x)**2,(-6*np.array([1,1,1]),6*np.array([1,1,1])),np.array([0,0,0]),2,BATCH_SIZE*steps,presteps=10)
		# 	indx = torch.randperm(steps*BATCH_SIZE)
		#X_all.requires_grad = True
		
		for step in range(steps):
			if (epoch)%4 == 1:
			
				symloss=torch.mean(((net(S_left)-net(S_right))/(net(S_left)+net(S_right)))**2)
			
				J     = symloss
				
			elif (epoch)%4 == 3:
			
				X = (torch.rand(BATCH_SIZE,3,requires_grad=True)-0.5)*2*5
				eps_0 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)
				eps_1 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)

				Psi_0_p = net(X+eps_0)
				Psi_0_n = net(X-eps_0)
				Psi_0   = (Psi_0_p + Psi_0_n)/2
				grad_0  = grad(Psi_0,X,create_graph=True,grad_outputs=torch.ones_like(Psi_0))[0]

				Psi_1_p = net(X+eps_1)
				Psi_1_n = net(X-eps_1)
				Psi_1   = (Psi_1_p + Psi_1_n)/2
				grad_1  = grad(Psi_1,X,create_graph=True,grad_outputs=torch.ones_like(Psi_1))[0]
				
				r1    = torch.norm(X-R1,dim=1)
				r2    = torch.norm(X-R2,dim=1)
				V     = -1/r1 - 1/r2 #+ 1/R  # is constant offset that does not influence the fitting procedure
				
				gradloss = torch.sum(0.5*torch.sum(grad_0*grad_1,dim=1)+Psi_0*V*Psi_1)/torch.sum(Psi_0*Psi_1)
			
				J = gradloss
			
			else:
				
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
				V     = -1/r1 - 1/r2 #+ 1/R
			
				laploss  = torch.mean(((-Lap_0/Psi_0 + V-net.Lambda)*(-Lap_1/Psi_1+ V-net.Lambda)))
				
				J = laploss
			 
			# w     = Psi_0*Psi_1/Psi_t**2
			# J     = torch.sum(w*(-Lap_0/Psi_0 + (V-net.Lambda))*(-Lap_1/Psi_1+ (V-net.Lambda)))/torch.sum(Psi_0*Psi_1/Psi_t)
			
			
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
		E     = torch.mean(torch.sum(gPsi**2,dim=1)/2+Psi**2*V)/torch.mean(Psi**2) # should give ~ -0.6023424 (-16.4) for hydrogen ion at (R ~ 2 a.u.)


		print('___________________________________________')
		print('It took', time.time()-start, 'seconds.')
		print('Lambda = '+str(net.Lambda[0].item()*27.2))
		print('Energy = '+str(E.item()*27.2))
		print('Alpha  = '+str(net.alpha[0].item()))
		print('Beta   = '+str(net.beta[0].item()))
		print('\n')


		X_plot = torch.from_numpy(np.swapaxes(np.array([np.linspace(-6,6,100),np.zeros(100),np.zeros(100)]).reshape(3,100),0,1)).type(torch.FloatTensor)
		Psi_plot = net(X_plot).detach().numpy()
		if Psi_plot[np.argmax(np.abs(Psi_plot))] < 0:
			Psi_plot *= -1
		plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E.item()*27.2,2)),ls=':',color=cmap(epoch/epochs))


	X_plot = torch.from_numpy(np.swapaxes(np.array([np.linspace(-6,6,100),np.zeros(100),np.zeros(100)]).reshape(3,100),0,1)).type(torch.FloatTensor)
	Psi_plot = net(X_plot).detach().numpy()
	if Psi_plot[np.argmax(np.abs(Psi_plot))] < 0:
		Psi_plot *= -1
	plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E.item()*27.2,2)),color='r')


plt.axvline(R1.numpy()[0],ls=':',color='k')
plt.axvline(R2.numpy()[0],ls=':',color='k')

plt.legend(loc="upper right")
plt.savefig("test.png")
plt.show()
