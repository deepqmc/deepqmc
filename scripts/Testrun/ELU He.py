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
from HMC import * 

cmap=plt.get_cmap("plasma")



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



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
		pro = torch.zeros(6)
		pro[0:3] = (torch.rand(3)-0.5)*maxstepsize
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

def fit(batch_size=2056,steps=15,epochs=4,losses=["variance","energy","symmetry"]):
	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.NN=nn.Sequential(
					torch.nn.Linear(3, 64),
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

			
		def forward(self,x):                #define forward pass
			d = torch.zeros(len(x),3)       #get distances
			d[:,0] = torch.norm(x[:,:3],dim=1)
			d[:,1] = torch.norm(x[:,3:],dim=1)
			d[:,2] = torch.norm(x[:,:3]-x[:,3:],dim=1)
			d2 = torch.zeros(len(x),3)  
			d2[:,0] = d[:,1]
			d2[:,1] = d[:,0]
			d2[:,2] = d[:,2]
			r = torch.erf(d/0.01)/d         #get inverse distances
			r2 = torch.erf(d2/0.01)/d2
			return self.NN(r)[:,0]+self.NN(r2)[:,0]


	LR=1e-4

	limits = 10
	points = 300
	X_plot = torch.from_numpy(np.swapaxes(np.array([np.linspace(-limits,limits,points),np.zeros(points),np.zeros(points),1*np.ones(points),np.zeros(points),np.zeros(points)]).reshape(6,points),0,1)).type(torch.FloatTensor)

	net = Net()
	params = [p for p in net.parameters()]
	opt = torch.optim.Adam(params, lr=LR)
	E = 100
	E_min = 100


	plt.figure(figsize=(12,9))
	plt.subplots_adjust(bottom=0.3)
	for epoch in range(epochs):

		savenet = (copy.deepcopy(net),E)

		print('___________________________________________')
		print('___________________________________________')
		print("epoch " +str(1+epoch)+" of "+str(epochs)+":")

		if losses[epoch%len(losses)] == "energy":
			print("minimize energy")
		elif losses[epoch%len(losses)] == "variance":
			print("minimize variance of energy")
		else:
			print("loss error, check losses:"+str(losses[epoch%len(losses)] ))
		start = time.time()



		X_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,6))*1).type(torch.FloatTensor)

		#X1_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
		#X2_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
		#X3_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
		#X4_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
		#X5_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
		#X6_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
		#XS_all  = torch.cat([X1_all,X2_all,X3_all], dim=0).reshape(3,batch_size*steps).transpose(0,1)

		index = torch.randperm(steps*batch_size)
		X_all.requires_grad = True
		# X1_all.requires_grad = True
		# X2_all.requires_grad = True
		# X3_all.requires_grad = True
		# X4_all.requires_grad = True
		# X5_all.requires_grad = True
		# X6_all.requires_grad = True

		for step in range(steps):

			if losses[epoch%len(losses)] == "energy":

				X = X_all[index[step*batch_size:(step+1)*batch_size]]
				#grad_X = grad(Psi,X,create_graph=True,grad_outputs=torch.ones_like(Psi))[0]
				#gradloss = torch.sum(0.5*torch.sum(grad_X**2,dim=1)+Psi*V*Psi)/torch.sum(Psi**2)


				r1    = torch.norm(X[:,:3],dim=1)
				r2    = torch.norm(X[:,3:],dim=1)
				r3    = torch.norm(X[:,:3]-X[:,3:],dim=1)
				V     = -2/r1 - 2/r2 + 2/r3
				Psi=net(X).flatten()

				g =torch.autograd.grad(Psi,X,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
				gradloss  = torch.mean(0.5*(torch.sum(g**2,dim=1)) + Psi**2*(V))/torch.mean(Psi**2)

				J = gradloss + (torch.sum(Psi**2)-1)**2

			# elif losses[epoch%len(losses)] == "variance":
			#
			# 	X1 = X1_all[index[step*batch_size:(step+1)*batch_size]]
			# 	X2 = X2_all[index[step*batch_size:(step+1)*batch_size]]
			# 	X3 = X3_all[index[step*batch_size:(step+1)*batch_size]]
			# 	X4 = X4_all[index[step*batch_size:(step+1)*batch_size]]
			# 	X5 = X5_all[index[step*batch_size:(step+1)*batch_size]]
			# 	X6 = X6_all[index[step*batch_size:(step+1)*batch_size]]
			# 	X=torch.cat([X1,X2,X3,X4,X5,X6], dim=0).reshape(6,batch_size).transpose(0,1)
			#
			# 	Psi=net(X).flatten()
			# 	dx1 =torch.autograd.grad(Psi,X1,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
			# 	ddx1=torch.autograd.grad(dx1[0].flatten(),X1,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
			# 	dy1 =torch.autograd.grad(Psi,X2,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
			# 	ddy1=torch.autograd.grad(dy1[0].flatten(),X2,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
			# 	dz1 =torch.autograd.grad(Psi,X3,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
			# 	ddz1=torch.autograd.grad(dz1[0].flatten(),X3,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
			# 	dx2 =torch.autograd.grad(Psi,X4,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
			# 	ddx2=torch.autograd.grad(dx2[0].flatten(),X4,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
			# 	dy2 =torch.autograd.grad(Psi,X5,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
			# 	ddy2=torch.autograd.grad(dy2[0].flatten(),X5,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
			# 	dz2 =torch.autograd.grad(Psi,X6,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
			# 	ddz2=torch.autograd.grad(dz2[0].flatten(),X6,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
			# 	lap_X = (ddx1+ddy1+ddz1+ddx2+ddy2+ddz2).flatten()
			#
			#
			# 	r1    = torch.norm(X[:,:3]-R1,dim=1)
			# 	r2    = torch.norm(X[:,:3]-R2,dim=1)
			# 	r3    = torch.norm(X[:,3:]-R1,dim=1)
			# 	r4    = torch.norm(X[:,3:]-R2,dim=1)
			# 	r5    = torch.norm(X[:,:3]-X[:,3:],dim=1)
			# 	V     = -1/r1 - 1/r2 - 1/r3 - 1/r4 + 1/r5
			#
			#
			#
			# 	#laploss  = torch.sqrt(torch.sum((-0.5*lap_X*Psi + (V-net.Lambda)*Psi**2)**2))*1e1
			#
			#
			# 	#gradloss  = torch.sum(0.5*(dx[0]**2+dy[0]**2+dz[0]**2) + Psi**2*(V))
			# 	laploss  = torch.sum(Psi*(-0.5*lap_X + Psi*V)) * 11000
			# 	#print(gradloss/laploss)
			#
			#
			#
			# 	#laploss  = torch.sum((-0.5*lap_X + (V-net.Lambda)*Psi)**2/(Psi**2))#/torch.sum(Psi**2)
			#
			# 	J = laploss + (torch.sum(Psi**2)-1)**2

			opt.zero_grad()
			J.backward()
			opt.step()


			print("Progress {:2.0%}".format(step /steps), end="\r")


		#compute energy of wavefunction after each epoch
		E_mean = 0     #initialise mean
		E_square = 0   #initialise square (for var)
		ex = 4	   #exponent for number of steps of metropolis algorithm
		n_mean = 3	   #number of times the metropolis algorithm is performt (to indicate convergence)
			
		for i,T in enumerate([0.1 * (i+1) for i in range(3)]):

			ts = time.time()

			samples = HMC(net,0.5,2,10,1000,6,T=T) #obtain samples
			samples = samples.view(-1,6).detach()

			
			X1 = samples[:,0]
			X2 = samples[:,1]
			X3 = samples[:,2]
			X4 = samples[:,3]
			X5 = samples[:,4]
			X6 = samples[:,5]
			X1.requires_grad=True
			X2.requires_grad=True
			X3.requires_grad=True
			X4.requires_grad=True
			X5.requires_grad=True
			X6.requires_grad=True
			X=torch.cat([X1,X2,X3,X4,X5,X6], dim=0).reshape(6,10000).transpose(0,1)
			Psi=net(X).flatten()

			dx1 =torch.autograd.grad(Psi,X1,create_graph=True,retain_graph=True,grad_outputs=torch.ones(len(X1)))
			ddx1=torch.autograd.grad(dx1[0].flatten(),X1,retain_graph=True,grad_outputs=torch.ones(len(X1)))[0]
			dy1 =torch.autograd.grad(Psi,X2,create_graph=True,retain_graph=True,grad_outputs=torch.ones(len(X1)))
			ddy1=torch.autograd.grad(dy1[0].flatten(),X2,retain_graph=True,grad_outputs=torch.ones(len(X1)))[0]
			dz1 =torch.autograd.grad(Psi,X3,create_graph=True,retain_graph=True,grad_outputs=torch.ones(len(X1)))
			ddz1=torch.autograd.grad(dz1[0].flatten(),X3,retain_graph=True,grad_outputs=torch.ones(len(X1)))[0]
			dx2 =torch.autograd.grad(Psi,X4,create_graph=True,retain_graph=True,grad_outputs=torch.ones(len(X1)))
			ddx2=torch.autograd.grad(dx2[0].flatten(),X4,retain_graph=True,grad_outputs=torch.ones(len(X1)))[0]
			dy2 =torch.autograd.grad(Psi,X5,create_graph=True,retain_graph=True,grad_outputs=torch.ones(len(X1)))
			ddy2=torch.autograd.grad(dy2[0].flatten(),X5,retain_graph=True,grad_outputs=torch.ones(len(X1)))[0]
			dz2 =torch.autograd.grad(Psi,X6,create_graph=True,retain_graph=True,grad_outputs=torch.ones(len(X1)))
			ddz2=torch.autograd.grad(dz2[0].flatten(),X6,retain_graph=True,grad_outputs=torch.ones(len(X1)))[0]
			lap_X = (ddx1+ddy1+ddz1+ddx2+ddy2+ddz2).flatten()

			r1    = torch.norm(X[:,:3],dim=1)
			r2    = torch.norm(X[:,3:],dim=1)
			r3    = torch.norm(X[:,:3]-X[:,3:],dim=1)
			V     = -2/r1 - 2/r2 + 1/r3

			E_loc_lap = -0.5*lap_X/Psi

			E    = (torch.mean(E_loc_lap + V).item()*27.211386) # energy is given by mean of local energy over sampled batch from psi**2
			print('#samples = '+str(10000)+'    energyexpextation = '+str(E)+'    time = '+str(np.round((time.time()-ts),2)))
			E_mean += E
			E_square += E**2
			
			#print(X1)
			plt.figure()
			plt.hist(X1.detach().numpy(),density=True,bins=100)
			Psi=net(X_plot).detach().numpy()
			plt.plot(X_plot[:,0].detach().numpy(),Psi**2/np.max(Psi**2))
			plt.title("Energyexpectation = "+str(E))
			plt.savefig(datetime.datetime.now().strftime("%B%d%Y%I%M%p")+".png")

		E = E_mean/n_mean



		print('It took', time.time()-start, 'seconds.')
		#print('Lambda = '+str(net.Lambda[0].item()*27.2))
		print('Energy = '+str(E_mean/n_mean)+' +- '+str((E_square/n_mean-(E_mean/n_mean)**2)**(1/2)))

		print('\n')



		if False:#else:

			Psi_plot = net(X_plot).detach().numpy()
			#Psi_plot2 = net(X_plot2).detach().numpy()
			if Psi_plot[np.argmax(np.abs(Psi_plot))] < 0:
				print("negative Psi")
				#Psi_plot *= -1

			if epoch<(epochs-1) :
					plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),ls=':',color=cmap(epoch/epochs),linewidth =2)
					#plt.plot(X_plot[:,0].numpy(),(Psi_plot2/max(np.abs(Psi_plot2)))**2,label=str(np.round(E,2)),ls=':',color=cmap(epoch/epochs),linewidth =2)

			else:
				plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),color='k',linewidth =3)
				#plt.plot(X_plot[:,0].numpy(),(Psi_plot2/max(np.abs(Psi_plot2)))**2,label=str(np.round(E,2)),color='k',linewidth =2)



	plt.legend(loc="lower center",bbox_to_anchor=[0.5, - 0.4], ncol=8)
	#plt.savefig(datetime.datetime.now().strftime("%B%d%Y%I%M%p")+".png")
	#plt.show()
	#return (Psi_min,E_min)
#	fig = plt.figure()
#	ax = fig.add_subplot(1, 1, 1, projection='3d')
#	ax.scatter(X1.detach().numpy(), X2.detach().numpy(), X3.detach().numpy(), c=np.abs(Psi.detach().numpy()), cmap=plt.hot())
#	plt.show()
for i in range(5):
	fit(batch_size=1500,steps=5,epochs=5,losses=["energy"])
plt.show()
