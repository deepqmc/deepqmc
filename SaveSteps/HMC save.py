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

from pynverse import inversefunc

from scipy.stats import norm


def metropolis(distribution,startpoint,maxstepsize,steps,presteps=0,interval=None):
	#initialise list to store walker positions
	samples = torch.zeros(steps,len(startpoint))
	#another list for the ratios
	ratios = torch.zeros(steps)
	#initialise the walker at the startposition
	walker = torch.tensor([startpoint]).type(torch.FloatTensor)
	distwalker = distribution(walker)
	#loop over proposal steps
	for i in range(presteps+steps):
		#append position of walker to the sample list in case presteps exceeded
		if i > (presteps-1):
			samples[i-presteps]=(walker)
		#propose new trial position
		#trial = walker + (torch.rand(6)-0.5)*maxstepsize
		#pro = torch.zeros(2)
		pro = (torch.rand(2)-0.5)*maxstepsize
		trial = walker + pro
		#calculate acceptance propability
		disttrial = distribution(trial)
		#check if in interval
		if not interval is None:
			inint = torch.tensor(all(torch.tensor(interval[0]).type(torch.FloatTensor)<trial[0]) \
			and all(torch.tensor(interval[1]).type(torch.FloatTensor)>trial[0])).type(torch.FloatTensor)
			disttrial = disttrial*inint

		ratio = disttrial/distwalker
		ratios[i-presteps] = ratio
		#accept trial position with respective propability
		if ratio > np.random.uniform(0,1):
			walker = trial
			distwalker = disttrial
	#return list of samples
	print("variance of acc-ratios = " + str((torch.sqrt(torch.mean(ratios**2)-torch.mean(ratios)**2)).data))
	return samples


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


def get_grad_Psisquare(net,pos):
	pos.requires_grad=True
	Psi=net(pos)
	g_Psi =torch.autograd.grad(Psi,pos,create_graph=True,\
	retain_graph=True,grad_outputs=torch.ones(len(pos)))[0]
	g_Psisquare = 2*g_Psi*Psi[:,None]
	return g_Psisquare

def dynamics(net,pos,stepsize,steps,T):
	vel = torch.randn(pos.shape)*T
	v_te2 = vel + stepsize*get_grad_Psisquare(net,pos.detach())*(-1)/2
	p_te  = pos + stepsize*v_te2
	for n in range(1,steps):
		v_te2 = v_te2 - stepsize*get_grad_Psisquare(net,p_te.detach())*(-1)
		p_te  = p_te + stepsize*v_te2
	v_te  = v_te2 - stepsize/2*get_grad_Psisquare(net,p_te.detach())*(-1)
	return p_te,vel.detach().numpy()
	
	
def HMC(net,stepsize,steps,n_walker,n_samples,dim,startfactor=0.7,T=0.3,presteps=100):
	samples = torch.zeros(n_samples,n_walker,dim)
	walker = torch.randn(n_walker,dim)*startfactor
	v_walker = np.random.normal(size=(n_walker,dim))*startfactor
	distwalker = net(walker).detach().numpy()
	for i in range(n_samples+presteps):
		if i>=presteps:
			samples[i-presteps] = walker
		trial,v_trial = dynamics(net,walker,stepsize,steps,T)
		disttrial = net(trial).detach().numpy()
		ratio = (np.exp(disttrial)/np.exp(distwalker))*(np.exp(0.5*np.sum(v_trial**2,axis=-1))/np.exp(0.5*np.sum(v_walker**2,axis=-1)))
		smaller_n = (ratio<np.random.uniform(0,1,n_walker)).astype(float)
		larger_n  = np.abs(smaller_n-1)
		smaller = torch.from_numpy(smaller_n).type(torch.FloatTensor)
		larger  = torch.abs(smaller-1)
		walker = trial*larger[:,None] + walker*smaller[:,None]
		v_walker = v_trial*larger_n[:,None] + v_walker*smaller_n[:,None]

	return samples

POS=np.array(HMC(net,0.1,10,20,1000,2,T=0.1).detach())
POS2 = np.array(metropolis(lambda x: net(x)**2,np.array([1,1]),0.5,20000,presteps=100,interval=None).detach())


plt.figure(figsize=(18,5))
plt.subplot2grid((1,3),(0,0))
plt.imshow(P.detach().numpy(),extent=[-2,2,-2,2],cmap=cmap)

#for i in range(10):
#	plt.plot(POS[:,i,0],POS[:,i,1],ls='',Marker='.',ms=1)

plt.subplot2grid((1,3),(0,1))
plt.hist2d(POS[:,:,0].flatten(),POS[:,:,1].flatten(),range=np.array([[-2,2],[-2,2]]),cmap=cmap,bins=20)
plt.subplot2grid((1,3),(0,2))
plt.hist2d(POS2[:,0].flatten(),POS2[:,1].flatten(),range=np.array([[-2,2],[-2,2]]),cmap=cmap,bins=20)


plt.show()
exit(0)

