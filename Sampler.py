import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import torch
import torch.nn as nn

import copy
import scipy as sp
from scipy.stats import norm

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
	#logdist = lambda x: -torch.log(dist(x))
	acc=0
	samples = torch.zeros(steps,n_walker,dim)
	walker = torch.randn(n_walker,dim)*startfactor
	#plt.plot(walker.detach().numpy()[:,0],walker.detach().numpy()[:,1],marker='.',ms='1',ls='',color='k')
	v_walker = np.zeros((n_walker,dim))
	#distwalker = logdist(walker).detach().numpy()
	distwalker = dist(walker).detach().numpy()
	for i in range(steps+presteps):
		if i>=presteps:
			samples[i-presteps] = walker
		#trial,v_trial = dynamics(logdist,walker,stepsize,dysteps,push)
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
		acc += np.sum(larger_n)
		#print(v_walker,larger)
		#v_walker = v_trial*larger_n[:,None] + v_walker*smaller_n[:,None]
	print(acc/(n_walker*steps))

	return samples


def metropolis(distribution,startpoint,stepsize,steps,dim,n_walker,startfactor=0.2,presteps=0,interval=None,T=0.2):
	#initialise list to store walker positions
	samples = torch.zeros(steps,n_walker,len(startpoint))
	#another list for the ratios
	ratios = np.zeros((steps,n_walker))
	#initialise the walker at the startposition
	walker = torch.randn(n_walker,dim)*startfactor
	#plt.plot(walker.detach().numpy()[:,0],walker.detach().numpy()[:,1],marker='.',ms='1',ls='',color='k')
	distwalker = distribution(walker)
	#loop over proposal steps
	for i in range(presteps+steps):
		#append position of walker to the sample list in case presteps exceeded
		if i > (presteps-1):
			samples[i-presteps]=walker
		#propose new trial position
		#trial = walker + (torch.rand(6)-0.5)*maxstepsize
		#pro = torch.zeros(2)
		pro = (torch.rand(walker.shape)-0.5)*stepsize
		trial = walker + pro
		#calculate acceptance propability
		disttrial = distribution(trial)
		#check if in interval
		if not interval is None:
			inint = torch.tensor(all(torch.tensor(interval[0]).type(torch.FloatTensor)<trial[0]) \
			and all(torch.tensor(interval[1]).type(torch.FloatTensor)>trial[0])).type(torch.FloatTensor)
			disttrial = disttrial*inint

		ratio = np.exp((disttrial.detach().numpy()-distwalker.detach().numpy())/T)
		ratios[i-presteps] = ratio
		#accept trial position with respective propability
		smaller_n = (ratio<np.random.uniform(0,1,n_walker)).astype(float)
		larger_n  = np.abs(smaller_n-1)
		smaller = torch.from_numpy(smaller_n).type(torch.FloatTensor)
		larger  = torch.abs(smaller-1)
		walker = trial*larger[:,None] + walker*smaller[:,None]

		#if ratio > np.random.uniform(0,1):
	#		walker = trial
	#		distwalker = disttrial
	#return list of samples
	#print("variance of acc-ratios = " + str((np.sqrt(np.mean(ratios**2)-np.mean(ratios)**2)).data))
	return samples
