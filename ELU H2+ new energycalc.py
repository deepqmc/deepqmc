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


cmap=plt.get_cmap("plasma")

def metropolis(distribution,interval,startpoint,maxstepsize,steps,presteps=0):
	#initialise list to store walker positions
	samples = torch.zeros(steps,len(startpoint))
	#initialise the walker at the startposition
	walker = torch.tensor([startpoint]).type(torch.FloatTensor)
	distwalker = distribution(walker)
	#loop over proposal steps
	for i in range(presteps+steps):
		#append position of walker to the sample list in case presteps exceeded
		if i > (presteps-1):
			samples[i-presteps]=(walker)
		#propose new trial position
		trial = walker + (torch.rand(3)-0.5)*maxstepsize
		#check if in interval
		inint = torch.tensor(all(torch.tensor(interval[0]).type(torch.FloatTensor)<trial[0]) \
		and all(torch.tensor(interval[1]).type(torch.FloatTensor)>trial[0])).type(torch.FloatTensor)
		#calculate acceptance propability
		disttrial = distribution(trial)*inint
		ratio = disttrial/distwalker
		#accept trial position with respective propability
		if ratio > np.random.uniform(0,1):
			walker = trial
			distwalker = disttrial
	#return list of samples
	return samples

def fit(batch_size=2056,steps=15,epochs=4,R1=1.5,R2=-1.5,losses=["variance","energy","symmetry"]):
	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.NN=nn.Sequential(
					torch.nn.Linear(2, 64),
					torch.nn.ELU(),
					torch.nn.Linear(64, 64),
					torch.nn.ELU(),
					torch.nn.Linear(64, 64),
					#torch.nn.ELU(),
					#torch.nn.Linear(64, 64),
					torch.nn.ELU(),
					torch.nn.Linear(64, 1)
					)

			self.Lambda=nn.Parameter(torch.Tensor([-1]))	#energy eigenvalue
			self.alpha=nn.Parameter(torch.Tensor([1]))#coefficient for decay
			self.beta=nn.Parameter(torch.Tensor([8]))#coefficient for decay

		def forward(self,x):
			d = torch.zeros(len(x),2)
			d[:,0] = torch.norm(x-R1,dim=1)
			d[:,1] = torch.norm(x-R2,dim=1)
			return (self.NN(d)[:,0])*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))

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

	LR=5e-3

	R1    = torch.tensor([R1,0,0]).type(torch.FloatTensor)
	R2    = torch.tensor([R2,0,0]).type(torch.FloatTensor)
	R     = torch.norm(R1-R2)

	#X_plot = torch.from_numpy(np.swapaxes(np.array([np.linspace(-6,6,100),np.zeros(100),np.zeros(100)]).reshape(3,100),0,1)).type(torch.FloatTensor)

	net = Net()
	net.alpha=nn.Parameter(torch.Tensor([6/R]))
	params = [p for p in net.parameters()]
	#del params[0]
	#del params[1]
	#del params[1]
	opt = torch.optim.Adam(params, lr=LR)
	E = 100
	E_min = 100

	for epoch in range(epochs):

		savenet = (copy.deepcopy(net),E)


		print("epoch " +str(1+epoch)+" of "+str(epochs)+":")
		if losses[epoch%len(losses)] == "symmetry":
			print("symmetrize")
		elif losses[epoch%len(losses)] == "energy":
			print("minimize energy")
		elif losses[epoch%len(losses)] == "variance":
			print("minimize variance of energy")
		else:
			print("loss error, check losses:"+str(losses[epoch%len(losses)] ))
		start = time.time()


		X_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,3))*(3*R/2).numpy()).type(torch.FloatTensor)
		#X1_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*(R/2).numpy()).type(torch.FloatTensor)
		#X2_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*(R/2).numpy()).type(torch.FloatTensor)
		#X3_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*(R/2).numpy()).type(torch.FloatTensor)
		#X_all=torch.cat([X1_all,X2_all,X3_all], dim=0).reshape(3,batch_size*steps).transpose(0,1)
		if losses[epoch%len(losses)] == "variance":
			samples_all =metropolis(lambda x :net(x)**2,(-6*np.array([1,1,1]),6*np.array([1,1,1])),np.array([0,0,0]),2,batch_size*steps,presteps=100)
			X1_all = samples_all[:,0]
			X2_all = samples_all[:,1]
			X3_all = samples_all[:,2]
			X1_all.requires_grad=True
			X2_all.requires_grad=True
			X3_all.requires_grad=True

		index = torch.randperm(steps*batch_size)
		X_all.requires_grad = True
		#X1_all.requires_grad = True
		#X2_all.requires_grad = True
		#X3_all.requires_grad = True




		for step in range(steps):


			if losses[epoch%len(losses)] == "symmetry":

				symloss=torch.mean(((net(S_left)-net(S_right))/(net(S_left)+net(S_right)))**2)

				J     = symloss


			elif losses[epoch%len(losses)] == "energy":
				#X1 = X1_all[index[step*batch_size:(step+1)*batch_size]]
				#X2 = X2_all[index[step*batch_size:(step+1)*batch_size]]
				#X3 = X3_all[index[step*batch_size:(step+1)*batch_size]]
				#X=torch.cat([X1,X2,X3], dim=0).reshape(3,batch_size).transpose(0,1)
				X=X_all[index[step*batch_size:(step+1)*batch_size]]
				Psi=net(X).flatten()
				grad_X = grad(Psi,X,create_graph=True,grad_outputs=torch.ones_like(Psi))[0]

				r1    = torch.norm(X-R1,dim=1)
				r2    = torch.norm(X-R2,dim=1)
				V     = -1/r1 - 1/r2
				gradloss = torch.sum(0.5*torch.sum(grad_X**2,dim=1)+Psi*V*Psi)#/torch.sum(Psi**2)

				J = gradloss + (torch.mean(Psi**2)-1)**2

			elif losses[epoch%len(losses)] == "variance":
				#n = 10000
				# samples=metropolis(lambda x :net(x)**2,(-6*np.array([1,1,1]),6*np.array([1,1,1])),np.array([0,0,0]),2,n,presteps=100)
				# X1 = samples[:,0]
				# X2 = samples[:,1]
				# X3 = samples[:,2]
				# X1.requires_grad=True
				# X2.requires_grad=True
				# X3.requires_grad=True
				#print(X1_all.shape,step*batch_size)
				X1 = X1_all[index[step*batch_size:(step+1)*batch_size]]
				X2 = X2_all[index[step*batch_size:(step+1)*batch_size]]
				X3 = X3_all[index[step*batch_size:(step+1)*batch_size]]
				X=torch.cat([X1,X2,X3], dim=0).reshape(3,batch_size).transpose(0,1)
				n = batch_size
				Psi=net(X).flatten()
				dx =torch.autograd.grad(Psi,X1,create_graph=True,retain_graph=True,grad_outputs=torch.ones(n))
				ddx=torch.autograd.grad(dx[0].flatten(),X1,retain_graph=True,grad_outputs=torch.ones(n))[0]
				dy =torch.autograd.grad(Psi,X2,create_graph=True,retain_graph=True,grad_outputs=torch.ones(n))
				ddy=torch.autograd.grad(dy[0].flatten(),X2,retain_graph=True,grad_outputs=torch.ones(n))[0]
				dz =torch.autograd.grad(Psi,X3,create_graph=True,retain_graph=True,grad_outputs=torch.ones(n))
				ddz=torch.autograd.grad(dz[0].flatten(),X3,retain_graph=True,grad_outputs=torch.ones(n))[0]
				lap_X = (ddx+ddy+ddz).flatten()

				r1    = torch.norm(X-R1,dim=1)
				r2    = torch.norm(X-R2,dim=1)
				V     = -1/r1 - 1/r2

				laploss = torch.mean(torch.min((-0.5*lap_X/Psi + V - net.Lambda)**2,torch.ones_like(Psi)*1))/torch.mean(Psi**2)
				J = laploss + (torch.mean(Psi**2)-1)**2

			opt.zero_grad()
			J.backward()
			opt.step()


			print("Progress {:2.0%}".format(step /steps), end="\r")
		t = time.time()
		G=torch.meshgrid([torch.linspace(-5,5,150),torch.linspace(-5,5,150),torch.linspace(-5,5,150)])
		x=G[0].flatten().view(-1,1)
		y=G[1].flatten().view(-1,1)
		z=G[2].flatten().view(-1,1)
		Xe = torch.cat((x, y, z), 1)
		Xe.requires_grad=True
		Psi   = net(Xe)
		gPsi  = grad(Psi,Xe,create_graph=True,grad_outputs=torch.ones(len(Xe)))[0]
		r1    = torch.norm(Xe-R1,dim=1)
		r2    = torch.norm(Xe-R2,dim=1)
		V     = -1/r1 - 1/r2 + 1/R
		E     = (torch.mean(torch.sum(gPsi**2,dim=1)/2+Psi**2*V)/torch.mean(Psi**2)).item()*27.211386 # should give ~ -0.6023424 (-16.4) for hydrogen ion at (R ~ 2 a.u.)
		print(E)
		print(time.time()-t)

		samples=metropolis(lambda x :net(x)**2,(-6*np.array([1,1,1]),6*np.array([1,1,1])),np.array([0,0,0]),2,20000,presteps=500)
		X1 = samples[:,0]
		X2 = samples[:,1]
		X3 = samples[:,2]
		X1.requires_grad=True
		X2.requires_grad=True
		X3.requires_grad=True
		X=torch.cat([X1,X2,X3], dim=0).reshape(3,20000).transpose(0,1)
		Psi=net(X).flatten()
		dx =torch.autograd.grad(Psi,X1,create_graph=True,retain_graph=True,grad_outputs=torch.ones(20000))
		ddx=torch.autograd.grad(dx[0].flatten(),X1,retain_graph=True,grad_outputs=torch.ones(20000))[0]
		dy =torch.autograd.grad(Psi,X2,create_graph=True,retain_graph=True,grad_outputs=torch.ones(20000))
		ddy=torch.autograd.grad(dy[0].flatten(),X2,retain_graph=True,grad_outputs=torch.ones(20000))[0]
		dz =torch.autograd.grad(Psi,X3,create_graph=True,retain_graph=True,grad_outputs=torch.ones(20000))
		ddz=torch.autograd.grad(dz[0].flatten(),X3,retain_graph=True,grad_outputs=torch.ones(20000))[0]
		lap_X = (ddx+ddy+ddz).flatten()

		r1    = torch.norm(X-R1,dim=1)
		r2    = torch.norm(X-R2,dim=1)
		V     = -1/r1 - 1/r2 + 1/R
		E     = (torch.mean(-0.5*lap_X/Psi + V).item()*27.211386)


		print('___________________________________________')
		print('It took', time.time()-start, 'seconds.')
		print('Lambda = '+str(net.Lambda[0].item()*27.211386))
		print('Energy = '+str(E))
		print('Alpha  = '+str(net.alpha[0].item()))
		print('Beta   = '+str(net.beta[0].item()))
		print('\n')


		if E < E_min:
			E_min = E
			Psi_min = copy.deepcopy(net)

		if epoch > 1 and E > (savenet[1]+10*(1-epoch/epochs)**2):
			print("undo step as E_new = "+str(E)+" E_old = "+str(savenet[1]))

			net = savenet[0]
			E   = savenet[1]

			params = [p for p in net.parameters()]
			del params[0]
			opt = torch.optim.Adam(params, lr=LR)

	return (Psi_min,E_min)

fit(batch_size=1000,steps=1000,epochs=5,losses=["energy"],R1=1,R2=-1)
#
# E_min=[]
# Psi_min=[]
# Rs = np.linspace(1,3,20)
# for R in Rs:
# 	psi,e = fit(batch_size=1000,steps=5000,epochs=30,losses=["energy","symmetry"],R1=R/2,R2=-R/2)
# 	E_min.append(e)
# 	Psi_min.append(psi)
#
# np.save("E_min_save",E_min)
#
# X = torch.from_numpy(np.swapaxes(np.array([np.linspace(-6,6,100),np.zeros(100),np.zeros(100)]).reshape(3,100),0,1)).type(torch.FloatTensor)
# plt.figure()
# for i,Psi in enumerate(Psi_min):
# 	Psi_plot = Psi(X).detach().numpy()
# 	plt.plot(X[:,0].detach().numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label="R = "+str(Rs[i]))
# plt.legend()
# plt.savefig("Psi_min_plot_save.png")
# plt.figure()
# plt.plot(Rs,E_min)
# plt.savefig("E_min_plot_save.png")
