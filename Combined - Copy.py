import numpy as np
import matplotlib.pyplot as plt
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
					torch.nn.ELU(),
					torch.nn.Linear(64, 64),
					#torch.nn.ReLU(),
					#torch.nn.Linear(64, 1)
					)
			self.Lambda=nn.Parameter(torch.Tensor([-1]))	#eigenvalue
			self.alpha=nn.Parameter(torch.Tensor([1]))#coefficient for decay
			self.beta=nn.Parameter(torch.Tensor([8]))#coefficient for decay
		def forward(self,x):
			d = torch.zeros(len(x),2)
			d[:,0] = torch.norm(x-R1,dim=1)
			d[:,1] = torch.norm(x-R2,dim=1)
			return self.NN(d)[:,0]*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))



	LR=1e-3
	H=0.2 #smoothing


	R1    = torch.tensor([R1,0,0]).type(torch.FloatTensor)
	R2    = torch.tensor([R2,0,0]).type(torch.FloatTensor)
	R     = torch.norm(R1-R2)

	X_plot = torch.from_numpy(np.swapaxes(np.array([np.linspace(-6,6,100),np.zeros(100),np.zeros(100)]).reshape(3,100),0,1)).type(torch.FloatTensor)

	net = Net()
	net.alpha=nn.Parameter(torch.Tensor([5/R]))
	params = [p for p in net.parameters()]
	del params[0]
	#del params[1]
	#del params[1]
	opt = torch.optim.Adam(params, lr=LR)
	E = 100


	plt.figure(figsize=(12,9))
	plt.subplots_adjust(bottom=0.3)
	for epoch in range(epochs):

		savenet = (copy.deepcopy(net),E)


		print("epoch " +str(1+epoch)+" of "+str(epochs)+":")

		if losses[epoch%len(losses)] == "energy":
			print("minimize energy")
		elif losses[epoch%len(losses)] == "variance":
			print("minimize variance of energy")
		else:
			print("loss error, check losses:"+str(losses[epoch%len(losses)] ))
		start = time.time()
		if epoch==10:
			with torch.no_grad():
			 	X_all = metropolis(lambda x :net(x)**2,(-6*np.array([1,1,1]),6*np.array([1,1,1])),np.array([0,0,0]),2,batch_size*steps,presteps=10)
		else:
			X_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,3))*3/2*R.numpy()).type(torch.FloatTensor)

		index = torch.randperm(steps*batch_size)
		X_all.requires_grad = True


		for step in range(steps):
			X=X_all[index[step*batch_size:(step+1)*batch_size]]
			#X = torch.from_numpy(np.random.normal(0,1,(batch_size,3))*3/2*R.numpy()).type(torch.FloatTensor)
			#X.requires_grad = True

			r1    = torch.norm(X-R1,dim=1)
			r2    = torch.norm(X-R2,dim=1)
			V     = -1/r1 - 1/r2 #+ 1/R
			Psi = net(X)

			if losses[epoch%len(losses)] == "energy":

				grad_X = grad(Psi,X,create_graph=True,grad_outputs=torch.ones_like(Psi))[0]
				gradloss = torch.sum(0.5*torch.sum(grad_X**2,dim=1)+Psi*V*Psi)/torch.sum(Psi**2)

				J = gradloss

			elif losses[epoch%len(losses)] == "variance":

				grad_X = grad(Psi,X,create_graph=True,grad_outputs=torch.ones_like(Psi))[0]
				lap_X  = torch.sum(grad(grad_X,X,create_graph=True,grad_outputs=torch.ones_like(grad_X))[0],dim=1)

				laploss  = torch.mean((-0.5*lap_X + Psi**(V-net.Lambda))**2)

				J = laploss

			# w     = Psi_0*Psi_1/Psi_t**2
			# J     = torch.sum(w*(-Lap_0/Psi_0 + (V-net.Lambda))*(-Lap_1/Psi_1+ (V-net.Lambda)))/torch.sum(Psi_0*Psi_1/Psi_t)


			opt.zero_grad()
			J.backward()
			opt.step()


			print("Progress {:2.0%}".format(step /steps), end="\r")



		G=torch.meshgrid([torch.linspace(-7,7,50),torch.linspace(-7,7,50),torch.linspace(-7,7,50)])
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
		E     = (torch.mean(torch.sum(gPsi**2,dim=1)/2+Psi**2*V)/torch.mean(Psi**2)).item()*27.2 # should give ~ -0.6023424 (-16.4) for hydrogen ion at (R ~ 2 a.u.)


		print('___________________________________________')
		print('It took', time.time()-start, 'seconds.')
		print('Lambda = '+str(net.Lambda[0].item()*27.2))
		print('Energy = '+str(E))
		print('Alpha  = '+str(net.alpha[0].item()))
		print('Beta   = '+str(net.beta[0].item()))
		print('\n')


		if E > (savenet[1]+5*(1-epoch/epochs)):#**2):
			print("undo step as E_new = "+str(E)+" E_old = "+str(savenet[1]))

			Psi_plot = net(X_plot).detach().numpy()
			if Psi_plot[np.argmax(np.abs(Psi_plot))] < 0:
				Psi_plot *= -1
			plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),ls=':',color='grey',linewidth =1)

			net = savenet[0]
			E   = savenet[1]

			params = [p for p in net.parameters()]
			del params[0]
			opt = torch.optim.Adam(params, lr=LR)

		else:

			Psi_plot = net(X_plot).detach().numpy()
			if Psi_plot[np.argmax(np.abs(Psi_plot))] < 0:
				Psi_plot *= -1

			if epoch<(epochs-1) :
					plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),ls=':',color=cmap(epoch/epochs),linewidth =2)

			else:
				plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),color='k',linewidth =3)

		#plt.hist(X_all.detach().numpy()[:,0],density=True)
		#plt.show()

	plt.axvline(R1.numpy()[0],ls=':',color='k')
	plt.axvline(R2.numpy()[0],ls=':',color='k')

	plt.title("batch_size = "+str(batch_size)+", steps = "+str(steps)+", epochs = "+str(epochs)+", R = "+str(R.item())+", losses = "+str(losses))
	plt.legend(loc="lower center",bbox_to_anchor=[0.5, - 0.4], ncol=8)
	#plt.savefig(datetime.datetime.now().strftime("%B%d%Y%I%M%p")+".png")
	plt.show()
	return (net,E)

fit(batch_size=512,steps=1000,epochs=6,R1=1,R2=-1,losses=["energy"])
