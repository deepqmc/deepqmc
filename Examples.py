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


cmap=plt.get_cmap("plasma")

def fit(batch_size=2056,steps=1500,epochs=4,R1=1.5,R2=-1.5):

	LR=5e-3

	RR    = torch.tensor([[R1,0,0],[R2,0,0]]).type(torch.FloatTensor)
	R1    = torch.tensor([R1,0,0]).type(torch.FloatTensor)
	R2    = torch.tensor([R2,0,0]).type(torch.FloatTensor)
	R     = torch.norm(R1-R2)

	#X_plot = torch.from_numpy(np.swapaxes(np.array([np.linspace(-6,6,100),np.zeros(100),np.zeros(100)]).reshape(3,100),0,1)).type(torch.FloatTensor)

	net = Net([2,20,20,20,1])
	opt = torch.optim.Adam(net.parameters(), lr=LR)

	E = 100

	for epoch in range(epochs):

		start = time.time()

		X_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,3))*(3*R/2).numpy()).type(torch.FloatTensor)
		index = torch.randperm(steps*batch_size)
		X_all.requires_grad = True


		for step in range(steps):

			X=X_all[index[step*batch_size:(step+1)*batch_size]]
			Psi=net(X,RR).flatten()
			grad_X = grad(Psi,X,create_graph=True,grad_outputs=torch.ones_like(Psi))[0]

			r1    = torch.norm(X-R1,dim=1)
			r2    = torch.norm(X-R2,dim=1)
			V     = -1/r1 - 1/r2
			gradloss = torch.sum(0.5*torch.sum(grad_X**2,dim=1)+Psi*V*Psi)#/torch.sum(Psi**2)

			J = gradloss + (torch.mean(Psi**2)-1)**2
			opt.zero_grad()
			J.backward()
			opt.step()

			print("Progress {:2.0%}".format(step /steps), end="\r")


		t = time.time()

		G=torch.meshgrid([torch.linspace(-5,5,30),torch.linspace(-5,5,30),torch.linspace(-5,5,30)])
		x=G[0].flatten().view(-1,1)
		y=G[1].flatten().view(-1,1)
		z=G[2].flatten().view(-1,1)
		Xe = torch.cat((x, y, z), 1)
		Xe.requires_grad=True
		Psi   = net(Xe,RR).flatten()
		gPsi  = grad(Psi,Xe,create_graph=True,grad_outputs=torch.ones(len(Xe)))[0]
		r1    = torch.norm(Xe-R1,dim=1)
		r2    = torch.norm(Xe-R2,dim=1)
		V     = -1/r1 - 1/r2 + 1/R
		E     = (torch.mean(torch.sum(gPsi**2,dim=1)/2+Psi**2*V)/torch.mean(Psi**2)).item()*27.211386 # should give ~ -0.6023424 (-16.4) for hydrogen ion at (R ~ 2 a.u.)


		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(x.detach().numpy(),y.detach().numpy(),Psi.detach().numpy())
		plt.show()

		print("E = "+str(E))
		print(time.time()-t)

		for j in [1,3,5]:
			#dysteps=j
			#sy=1/dysteps*0.05
			nw=100
			samples = HMC(lambda x :(net(x,RR).flatten())**2,0.1,j,n_walker=nw,steps=2000,dim=3,push=1,presteps=50).detach().reshape(-1,3)
			n=samples.shape[0]

			X1 = samples[:,0]
			X3 = samples[:,2]
			X2 = samples[:,1]
			plt.hist2d(X1.numpy(),X2.numpy(),range=np.array([[-5,5],[-5,5]]),bins=50)
			plt.show()
			X1.requires_grad=True
			X2.requires_grad=True
			X3.requires_grad=True
			X=torch.cat([X1,X2,X3], dim=0).reshape(3,n).transpose(0,1)
			Psi=net(X,RR).flatten()
			dx =torch.autograd.grad(Psi,X1,create_graph=True,retain_graph=True,grad_outputs=torch.ones(n))
			ddx=torch.autograd.grad(dx[0].flatten(),X1,retain_graph=True,grad_outputs=torch.ones(n))[0]
			dy =torch.autograd.grad(Psi,X2,create_graph=True,retain_graph=True,grad_outputs=torch.ones(n))
			ddy=torch.autograd.grad(dy[0].flatten(),X2,retain_graph=True,grad_outputs=torch.ones(n))[0]
			dz =torch.autograd.grad(Psi,X3,create_graph=True,retain_graph=True,grad_outputs=torch.ones(n))
			ddz=torch.autograd.grad(dz[0].flatten(),X3,retain_graph=True,grad_outputs=torch.ones(n))[0]
			lap_X = (ddx+ddy+ddz).flatten()

			r1    = torch.norm(X-R1,dim=1)
			r2    = torch.norm(X-R2,dim=1)
			V     = -1/r1 - 1/r2 + 1/R
			E     = torch.mean(-0.5*lap_X.view(nw,-1)/Psi.view(nw,-1) + V.view(nw,-1),dim=-1).detach().numpy()*27.211386#.item()*27.211386)
			mean = np.mean(E)
			var  = np.sqrt(np.mean((E-mean)**2))
			print("Mean = "+str(mean))
			print("var = "+str(var))


		print('___________________________________________')
		print('It took', time.time()-start, 'seconds.')
		print('Energy = '+str(mean))
		print('\n')


	return

fit(batch_size=1000,steps=300,epochs=5,R1=1,R2=-1)
#
