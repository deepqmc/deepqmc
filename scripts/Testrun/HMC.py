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
	
	
def HMC(net,stepsize,steps,n_walker,n_samples,dim,startfactor=0.7,T=0.2,presteps=100):
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

def checkconvergence(samples,net,n_test,eps=0.5):
	X = torch.randn(2*n_test,samples.shape[-1])
	Psiqou = net(X[::2])**2/net(X[1::2])**2
	X = X.detach().numpy()
	samples = samples.detach().numpy()
	for i in range(n_test):
		tmp1  = np.sum(np.sum((samples<X[i]+np.ones(samples.shape[-1])*eps)*(samples>X[i]-np.ones(samples.shape[-1])*eps),axis=-1)==samples.shape[-1])
		tmp2  = np.sum(np.sum((samples<X[i+1]+np.ones(samples.shape[-1])*eps)*(samples>X[i+1]-np.ones(samples.shape[-1])*eps),axis=-1)==samples.shape[-1])
		print(tmp1/tmp2)
	print(Psiqou)

