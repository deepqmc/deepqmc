import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import torch
from torch.autograd import grad

import copy
import datetime

from Sampler import *
from NN import *


def Potential(x,R,R_charges):
	"""returns the potential energy (coulomb). 
		input x is a tensor of shape [batchsize, dimensionality] (samples)
		input R is a tensor	of shape [#nuclei, dimesnionality?]"""
		
	dimension = x.shape[-1]
	batchsize = x.shape[0]

	d = 1/torch.norm((x[:,None,:]-R.repeat(1,dimension//R.shape[-1])[None,:,:]).view(batchsize,-1,R.shape[-1]),dim=-1)
	D = 1/torch.norm(R[:,None,:]-R[None,:,:],dim=-1)
	ind = np.arange(0,len(R))
	D[ind,ind] = 0
	
	P = torch.sum(D)/2 - torch.sum(d,dim=-1)
	
	if dimension//R.shape[-1] != 1:
		x = x.view(batchsize,-1,R.shape[-1])
		de = 1/torch.norm(x[:,:,None,:]-x[:,None,:,:],dim=-1)
		ind = np.arange(0,x.shape[-2])
		de[:,ind,ind] = 0
		P = P + torch.sum(de,dim=(-2,-1))/2
	
	return P
	
	
def Laplacian(x,R,net):

	dimension = x.shape[-1]
	batchsize = x.shape[0]
	
	X = []
	for i in range(dimension):
		X.append(x[:,i])
		X[i].requires_grad=True

	x=torch.cat(X, dim=0).reshape(dimension,batchsize).transpose(0,1)
	Psi=net(x,R).flatten()
	
	lap = 0
	for i in range(dimension):
		d  = torch.autograd.grad(Psi,X[i],create_graph=True,retain_graph=True,grad_outputs=torch.ones(batchsize))[0]
		dd = torch.autograd.grad(d,X[i],retain_graph=True,grad_outputs=torch.ones(batchsize))[0]
		lap += dd
	
	return lap,Psi
	
	
def Gradient(x,R,net):

	dimension = x.shape[-1]
	batchsize = x.shape[0]
	
	x.requires_grad=True
	Psi  = net(x,R).flatten()
	grad = torch.autograd.grad(Psi,x,create_graph=True,grad_outputs=torch.ones(batchsize))[0]
	
	return grad,Psi
	
def Gridgenerator(dim,sites,interval,zerodim):

		G = torch.meshgrid([torch.linspace(interval[0],interval[1],sites) for i in range(dim)])
		X = []
		for i in range(dim):
			X.append(G[i].flatten().view(-1,1))
			
		X = torch.cat(X,1)
		
		for i in range(zerodim):
			X = torch.cat((X.transpose(0,1).flatten(),torch.zeros(sites**(dim)))).view(-1,sites**(dim)).transpose(0,1)

		return X.contiguous()

	

cmap=plt.get_cmap("plasma")

def fit(batch_size=2056,n_el=1,steps=2500,epochs=4,RR = torch.tensor([[1,0,0],[-1,0,0]]).type(torch.FloatTensor),RR_charges=[1,1]):

	LR  = 5e-3
	
	inputdim = n_el*RR.shape[0]+np.sum(np.arange(0,n_el))
	net = WaveNet([inputdim,20,20,20,1])
	
	opt = torch.optim.Adam(net.parameters(), lr=LR)

	for epoch in range(epochs):

		X_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,3*n_el))*3).type(torch.FloatTensor)
		index = torch.randperm(steps*batch_size)

		for step in range(steps):

			X=X_all[index[step*batch_size:(step+1)*batch_size]]

			grad_X,Psi = Gradient(X,RR,net)

			V     = Potential(X,RR)
			
			loss = torch.mean(0.5*torch.sum(grad_X**2,dim=1)+ V*Psi**2)/torch.mean(Psi**2)

			J = loss #+ (torch.mean(Psi**2)-1)**2
			
				
			opt.zero_grad()
			J.backward()
			opt.step()

			print("Progress {:2.0%}".format(step /steps), end="\r")


		t = time.time()
		

		X = Gridgenerator(3*n_el,10,(-5,5),0)
		
		grad_X,Psi = Gradient(X,RR,net)
		
		

		V     = Potential(X,RR)
		
		
		E     = (torch.mean(torch.sum(grad_X**2,dim=1)/2+Psi**2*V)/torch.mean(Psi**2)).item()#*27.211386 

		
		X = Gridgenerator(2,80,(-5,5),3*n_el-2)
		#index = torch.LongTensor([0, 3, 2,1,4,5])
		#y = torch.zeros_like(X)
		#y[:,index] = X
		#X = y
		#print(X)
		Psi = net(X,RR).flatten()
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X[:,0].detach().numpy(),X[:,1].detach().numpy(),Psi.detach().numpy()**2,marker='.')

		plt.show()

		print("E = "+str(E))
		print("time = "+str(time.time()-t))	
		
		plt.figure(figsize=(10,3))

		for i,j in enumerate([1,3,5]):
			
	
			t = time.time()
			
			nw=50
			samples = HMC(lambda x :(net(x,RR).flatten())**2,0.1,j,n_walker=nw,steps=5000,dim=3*n_el,push=1,presteps=50).detach().reshape(-1,3*n_el)
			
			plt.subplot2grid((1,3),(0,i))
			plt.hist2d(samples[:,0].detach().numpy(),samples[:,1].detach().numpy(),bins=100,range=[[-5,5],[-5,5]])

			lap_X,Psi = Laplacian(samples,RR,net)
			V         = Potential(samples,RR)
			
			
			E     = torch.mean(-0.5*lap_X.view(nw,-1)/Psi.view(nw,-1) + V.view(nw,-1),dim=-1).detach().numpy()#*27.211386
			mean = np.mean(E)
			var  = np.sqrt(np.mean((E-mean)**2))
			print("Mean = "+str(mean))
			print("var = "+str(var))
			print("time = "+str(time.time()-t))

		plt.show()
		print('___________________________________________')
		print('\n')


	return

#H2+
fit(batch_size=10000,n_el=1,steps=500,epochs=5,RR=torch.tensor([[-1,0,0],[1.,0,0]]))
#H2
#fit(batch_size=10000,n_el=2,steps=100,epochs=5,RR=torch.tensor([[-0.7,0,0],[0.7,0,0]]))
#He+
#fit(batch_size=10000,n_el=2,steps=100,epochs=5,RR=torch.tensor([[-0.7,0,0],[0.7,0,0]]))
