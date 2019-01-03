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
		trial = walker + (torch.rand(len(startpoint))-0.5)*maxstepsize
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
					torch.nn.Linear(5, 64),
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
			self.alpha=nn.Parameter(torch.Tensor([8]))#coefficient for decay
			self.beta=nn.Parameter(torch.Tensor([2]))#coefficient for decay
		def forward(self,x):
			d = torch.zeros(len(x),5)
			d[:,0] = torch.norm(x[:,:3]-R1,dim=1)
			d[:,1] = torch.norm(x[:,:3]-R2,dim=1)
			d[:,2] = torch.norm(x[:,3:]-R1,dim=1)
			d[:,3] = torch.norm(x[:,3:]-R2,dim=1)
			d[:,4] = torch.norm(x[:,:3]-x[:,3:],dim=1)
			return (self.NN(d)[:,0])*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))

	LR=1e-3



	R1    = torch.tensor([R1,0,0]).type(torch.FloatTensor)
	R2    = torch.tensor([R2,0,0]).type(torch.FloatTensor)
	R     = torch.norm(R1-R2)

	X_plot = torch.from_numpy(np.swapaxes(np.array([np.linspace(-6,6,100),np.zeros(100),np.zeros(100),3*np.ones(100),np.zeros(100),np.zeros(100)]).reshape(6,100),0,1)).type(torch.FloatTensor)
	X_plot2 = torch.from_numpy(np.swapaxes(np.array([3*np.ones(100),np.zeros(100),np.zeros(100),np.linspace(-6,6,100),np.zeros(100),np.zeros(100)]).reshape(6,100),0,1)).type(torch.FloatTensor)

	net = Net()
	net.alpha=nn.Parameter(torch.Tensor([3/2/R]))
	params = [p for p in net.parameters()]
	#del params[0]
	del params[1]
	del params[1]
	opt = torch.optim.Adam(params, lr=LR)
	E = 100
	E_min = 100


	plt.figure(figsize=(12,9))
	plt.subplots_adjust(bottom=0.3)
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

		if False:
			with torch.no_grad():
			 	X_all = metropolis(lambda x :net(x)**2,(-6*np.array([1,1,1]),6*np.array([1,1,1])),np.array([0,0,0]),2,batch_size*steps,presteps=10)
			X1_all = X_all[:,0]
			X2_all = X_all[:,1]
			X3_all = X_all[:,2]
		else:
			X_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,6))*3/2*R.numpy()).type(torch.FloatTensor)
			#X1_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
			#X2_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
			#X3_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
			#X4_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
			#X5_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
			#X6_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3/2*R.numpy()).type(torch.FloatTensor)
			#XS_all  = torch.cat([X1_all,X2_all,X3_all], dim=0).reshape(3,batch_size*steps).transpose(0,1)

		index = torch.randperm(steps*batch_size)
		X_all.requires_grad = True
		#X1_all.requires_grad = True
		#X2_all.requires_grad = True
		#X3_all.requires_grad = True
		#X4_all.requires_grad = True
		#X5_all.requires_grad = True
		#X6_all.requires_grad = True

		for step in range(steps):

			if losses[epoch%len(losses)] == "energy":

				X = X_all[index[step*batch_size:(step+1)*batch_size]]
				#grad_X = grad(Psi,X,create_graph=True,grad_outputs=torch.ones_like(Psi))[0]
				#gradloss = torch.sum(0.5*torch.sum(grad_X**2,dim=1)+Psi*V*Psi)/torch.sum(Psi**2)


				r1    = torch.norm(X[:,:3]-R1,dim=1)
				r2    = torch.norm(X[:,:3]-R2,dim=1)
				r3    = torch.norm(X[:,3:]-R1,dim=1)
				r4    = torch.norm(X[:,3:]-R2,dim=1)
				r5    = torch.norm(X[:,:3]-X[:,3:],dim=1)
				V     = -1/r1 - 1/r2 - 1/r3 - 1/r4 + 1/r5
				Psi=net(X).flatten()

				g =torch.autograd.grad(Psi,X,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
				gradloss  = torch.sum(0.5*(torch.sum(g**2,dim=1)) + Psi**2*(V))

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



		# G=torch.meshgrid([torch.linspace(-5,5,10),torch.linspace(-5,5,10),torch.linspace(-5,5,10),torch.linspace(-5.5,5.5,10),torch.linspace(-5.5,5.5,10),torch.linspace(-5.5,5.5,10)])
		#
		# x1=G[0].flatten().view(-1,1)
		# y1=G[1].flatten().view(-1,1)
		# z1=G[2].flatten().view(-1,1)
		# x2=G[3].flatten().view(-1,1)
		# y2=G[4].flatten().view(-1,1)
		# z2=G[5].flatten().view(-1,1)
		# Xe = torch.cat((x1, y1, z1, x2, y2, z2), 1)
		# Xe.requires_grad=True
		# Psi   = net(Xe)
		# gPsi  = grad(Psi,Xe,create_graph=True,grad_outputs=torch.ones(len(Xe)))[0]
		# r1    = torch.norm(Xe[:,:3]-R1,dim=1)
		# r2    = torch.norm(Xe[:,:3]-R2,dim=1)
		# r3    = torch.norm(Xe[:,3:]-R1,dim=1)
		# r4    = torch.norm(Xe[:,3:]-R2,dim=1)
		# r5    = torch.norm(Xe[:,:3]-Xe[:,3:],dim=1)
		# V     = -1/r1 - 1/r2 + 1/R -1/r3 - 1/r4 + 1/r5
		#
		# E     = (torch.mean(torch.sum(gPsi**2,dim=1)/2+Psi**2*V)/torch.mean(Psi**2)).item()*27.2 # should give ~ -0.6023424 (-16.4) for hydrogen ion at (R ~ 2 a.u.)
		for m in [10000,25000,50000]:
			t = time.time()

			samples=metropolis(lambda x :net(x)**2,(-6*np.array([1,1,1,1,1,1]),6*np.array([1,1,1,1,1,1])),np.array([0,0,0,0,0,0]),2,m,presteps=500)
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
			X=torch.cat([X1,X2,X3,X4,X5,X6], dim=0).reshape(6,m).transpose(0,1)
			Psi=net(X).flatten()
			dx1 =torch.autograd.grad(Psi,X1,create_graph=True,retain_graph=True,grad_outputs=torch.ones(m))
			ddx1=torch.autograd.grad(dx1[0].flatten(),X1,retain_graph=True,grad_outputs=torch.ones(m))[0]
			dy1 =torch.autograd.grad(Psi,X2,create_graph=True,retain_graph=True,grad_outputs=torch.ones(m))
			ddy1=torch.autograd.grad(dy1[0].flatten(),X2,retain_graph=True,grad_outputs=torch.ones(m))[0]
			dz1 =torch.autograd.grad(Psi,X3,create_graph=True,retain_graph=True,grad_outputs=torch.ones(m))
			ddz1=torch.autograd.grad(dz1[0].flatten(),X3,retain_graph=True,grad_outputs=torch.ones(m))[0]
			dx2 =torch.autograd.grad(Psi,X4,create_graph=True,retain_graph=True,grad_outputs=torch.ones(m))
			ddx2=torch.autograd.grad(dx2[0].flatten(),X4,retain_graph=True,grad_outputs=torch.ones(m))[0]
			dy2 =torch.autograd.grad(Psi,X5,create_graph=True,retain_graph=True,grad_outputs=torch.ones(m))
			ddy2=torch.autograd.grad(dy2[0].flatten(),X5,retain_graph=True,grad_outputs=torch.ones(m))[0]
			dz2 =torch.autograd.grad(Psi,X6,create_graph=True,retain_graph=True,grad_outputs=torch.ones(m))
			ddz2=torch.autograd.grad(dz2[0].flatten(),X6,retain_graph=True,grad_outputs=torch.ones(m))[0]
			lap_X = (ddx1+ddy1+ddz1+ddx2+ddy2+ddz2).flatten()

			r1    = torch.norm(X[:,:3]-R1,dim=1)
			r2    = torch.norm(X[:,:3]-R2,dim=1)
			r3    = torch.norm(X[:,3:]-R1,dim=1)
			r4    = torch.norm(X[:,3:]-R2,dim=1)
			r5    = torch.norm(X[:,:3]-X[:,3:],dim=1)
			V     = -1/r1 - 1/r2 - 1/r3 - 1/r4 + 1/r5 + 1/R

			print(torch.mean(-0.5*lap_X/Psi + V).item()*27.211386)
			print(time.time()-t)

		print('___________________________________________')
		print('It took', time.time()-start, 'seconds.')
		print('Lambda = '+str(net.Lambda[0].item()*27.2))
		print('Energy = '+str(E))
		print('Alpha  = '+str(net.alpha[0].item()))
		print('Beta   = '+str(net.beta[0].item()))
		print('\n')


		# if E < E_min:
		# 	E_min = E
		# 	Psi_min = copy.deepcopy(net)
		#
		# if False:#epoch > 1 and E > (savenet[1]+10*(1-epoch/epochs)**2):
		# 	print("undo step as E_new = "+str(E)+" E_old = "+str(savenet[1]))
		#
		# 	Psi_plot = net(X_plot).detach().numpy()
		# 	if Psi_plot[np.argmax(np.abs(Psi_plot))] < 0:
		# 		print("negative Psi")
		# 	#	Psi_plot *= -1
		# 	plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),ls=':',color='grey',linewidth =1)
		#
		# 	net = savenet[0]
		# 	E   = savenet[1]
		#
		# 	params = [p for p in net.parameters()]
		# 	del params[0]
		# 	opt = torch.optim.Adam(params, lr=LR)
		#
	if True:#else:

		Psi_plot = net(X_plot).detach().numpy()
		Psi_plot2 = net(X_plot2).detach().numpy()
		if Psi_plot[np.argmax(np.abs(Psi_plot))] < 0:
			print("negative Psi")
			#Psi_plot *= -1

		if epoch<(epochs-1) :
				plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),ls=':',color=cmap(epoch/epochs),linewidth =2)
				plt.plot(X_plot[:,0].numpy(),(Psi_plot2/max(np.abs(Psi_plot2)))**2,label=str(np.round(E,2)),ls=':',color=cmap(epoch/epochs),linewidth =2)

		else:
			plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),color='k',linewidth =3)
			plt.plot(X_plot[:,0].numpy(),(Psi_plot2/max(np.abs(Psi_plot2)))**2,label=str(np.round(E,2)),color='k',linewidth =2)
		#plt.hist(X_all.detach().numpy()[:,0],density=True)
		#plt.show()

	#plt.axvline(R1.numpy()[0],ls=':',color='k')
	#plt.axvline(R2.numpy()[0],ls=':',color='k')

	#plt.title("batch_size = "+str(batch_size)+", steps = "+str(steps)+", epochs = "+str(epochs)+", R = "+str(R.item())+", losses = "+str(losses))
	#plt.legend(loc="lower center",bbox_to_anchor=[0.5, - 0.4], ncol=8)
	#plt.savefig(datetime.datetime.now().strftime("%B%d%Y%I%M%p")+".png")
	#plt.show()
	#return (Psi_min,E_min)
#	fig = plt.figure()
#	ax = fig.add_subplot(1, 1, 1, projection='3d')
#	ax.scatter(X1.detach().numpy(), X2.detach().numpy(), X3.detach().numpy(), c=np.abs(Psi.detach().numpy()), cmap=plt.hot())
#	plt.show()
fit(batch_size=250,steps=5000,epochs=10,losses=["energy"],R1=0.7,R2=-0.7)
plt.show()
