import numpy as np
from itertools import islice
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

from scipy.stats import norm
from pysgmcmc import *

from pysgmcmc.samplers.sghmc import SGHMC
#from pysgmcmc.samplers.sgld import SGLD
import matplotlib.animation as animation
import matplotlib
#matplotlib.use('Qt5Agg') #use Qt5 as backend, comment this line for default backend

from matplotlib import pyplot as plt
from matplotlib import animation

def pysgmcmcsampler(dist,sampler_cls=SGHMC, lr=0.5,startfactor=0.2,num_burn_in_steps=10,num_samples=50000):
	rand1 = torch.rand(1,requires_grad=True)
	#rand2 = torch.randn(1).requires_grad_(True)
	sampler = sampler_cls(params=(rand1,),lr=lr, negative_log_likelihood=dist)
	_ = [sample for sample, _ in islice(sampler, num_burn_in_steps)]
	samples = np.asarray([sample for sample, _ in islice(sampler, num_samples)])
	return samples

def dynamics(dist,pos,stepsize,steps,push):
	pos = pos.detach().clone()
	pos.requires_grad=True
	vel = torch.randn(pos.shape)*push
	v_te2 = vel - stepsize*torch.autograd.grad(dist(pos),pos,create_graph=True,retain_graph=True,grad_outputs=torch.ones(pos.shape[0]))[0]/2
	p_te  = pos + stepsize*v_te2
	for n in range(1,steps):
		v_te2 = v_te2 - stepsize*torch.autograd.grad(dist(pos),pos,create_graph=True,retain_graph=True,grad_outputs=torch.ones(pos.shape[0]))[0]
		p_te  = p_te + stepsize*v_te2
	v_te  = v_te2 - stepsize*torch.autograd.grad(dist(pos),pos,create_graph=True,retain_graph=True,grad_outputs=torch.ones(pos.shape[0]))[0]/2
	return p_te,vel.detach().numpy()


def HMC(dist,stepsize,dysteps,n_walker,steps,dim,push,startfactor=1,T=1,presteps=200):
	logdist = lambda x: -torch.log(dist(x))
	samples = torch.zeros(steps,n_walker,dim)
	walker = torch.randn(n_walker,dim)*startfactor
	#plt.plot(walker.detach().numpy()[:,0],walker.detach().numpy()[:,1],marker='.',ms='1',ls='',color='k')
	v_walker = np.zeros((n_walker,dim))
	#distwalker = logdist(walker).detach().numpy()
	distwalker = dist(walker).detach().numpy()
	for i in range(steps+presteps):
		if i>=presteps:
			samples[i-presteps] = walker
		trial,v_trial = dynamics((lambda x: -dist(x)),walker,stepsize,dysteps,push)
		#disttrial = logdist(trial).detach().numpy()
		disttrial = dist(trial).detach().numpy()
		#ratio = (np.exp(-disttrial+distwalker)/T)*(np.exp(-0.5/T*(np.sum(v_trial**2,axis=-1)-np.sum(v_walker**2,axis=-1))))
		ratio = (disttrial/distwalker)*(np.exp(-0.5*(np.sum(v_trial**2,axis=-1)-np.sum(v_walker**2,axis=-1))))
		smaller_n = (ratio<np.random.uniform(0,1,n_walker)).astype(float)
		larger_n  = np.abs(smaller_n-1)
		smaller = torch.from_numpy(smaller_n).type(torch.FloatTensor)
		larger  = torch.abs(smaller-1)
		walker = (trial*larger[:,None] + walker*smaller[:,None])
		#print(v_walker,larger)
		#v_walker = v_trial*larger_n[:,None] + v_walker*smaller_n[:,None]

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
			if type(x) == tuple:
				d = ((x[0]**2)**(1/2)).view(-1,1)
			else:
				d = torch.norm(x,dim=1).view(-1,1)              #define forward pass
			r = torch.erf(d/0.1)/d         #get inverse distances
			return self.NN(r)[:,0]

net=Net()

#def f(x):
#    return - (- x[0]**2 + 25)*3/500
#f = lambda x: -(- x[0]**2 + 25)*3/500
f = lambda x: (torch.exp(-1/2*(torch.norm(x,dim=-1)+2)**2)/np.sqrt(2*np.pi)+torch.exp(-1/2*(torch.norm(x,dim=-1)-2)**2)/np.sqrt(2*np.pi))/2
#f = lambda x: (- torch.norm(x,dim=-1)**2 + 25)*3/500
X = torch.linspace(-5,5,100).view(-1,1)


#print( ( lambda x: net(x)**2 )((torch.randn(1),)))

#pos = pysgmcmcsampler(f).flatten()

pos = HMC(f,0.1,5,50,5000,1,push=1,T=1).detach().numpy()
#print(pos)

def update_hist(num, pos, line):
	hist=np.histogram(pos[:num],range=[-5,5],density=True,bins=200)
	line.set_data(np.concatenate([hist[1],hist[0][:-1]]).reshape(2,-1))

	return line,

def update_line(num, data, line):
	line.set_data(data[..., num-10:num])
	return line,

fig1 = plt.figure()
data = np.array([pos.flatten() 	,0.1*np.ones(len(pos.flatten()))])
#l, = plt.plot([],ls='',marker='.')
q, = plt.plot([])
#plt.xlim(-5, 5)
#plt.ylim(0, 0.3)

line_ani = animation.FuncAnimation(fig1, update_hist, 50000, fargs=(pos, q),interval=1, blit=True)
#line_ani = animation.FuncAnimation(fig1, update_line, 5000, fargs=(data, l),interval=50, blit=True)

plt.plot(X.detach().numpy(),f(X).detach().numpy())
#plt.hist(pos.flatten(),density=True)
plt.show()
