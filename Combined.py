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

def fit(batch_size=2056,steps=15,epochs=4,R1=1.5,R2=-1.5,losses=["variance","energy","symmetry"]):
	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.NN=nn.Sequential(
					torch.nn.Linear(2, 64),
					torch.nn.ReLU(),
					torch.nn.Linear(64, 64),
					torch.nn.ReLU(),
					torch.nn.Linear(64, 64),
					torch.nn.ReLU(),
					torch.nn.Linear(64, 64),
					torch.nn.ReLU(),
					torch.nn.Linear(64, 1)
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


	net = Net()
	net.alpha=nn.Parameter(torch.Tensor([5/R]))
	params = [p for p in net.parameters()]
	#del params[0]
	#del params[1]
	#del params[1]
	opt = torch.optim.Adam(params, lr=LR)
	E = 100


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

		# with torch.no_grad():
		# 	X_all,Psi_t_all = metropolis(lambda x :net(x)**2,(-6*np.array([1,1,1]),6*np.array([1,1,1])),np.array([0,0,0]),2,batch_size*steps,presteps=10)
		# 	indx = torch.randperm(steps*batch_size)
		#X_all.requires_grad = True

		for step in range(steps):
			if losses[epoch%len(losses)] == "symmetry":

				symloss=torch.mean(((net(S_left)-net(S_right))/(net(S_left)+net(S_right)))**2)

				J     = symloss

			elif losses[epoch%len(losses)] == "energy":

				#X = (torch.rand(batch_size,3,requires_grad=True)-0.5)*2*5

				X = torch.from_numpy(np.random.normal(0,1,(batch_size,3))*3/2*R.numpy()).type(torch.FloatTensor)
				X.requires_grad = True

				eps_0 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)
				eps_1 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)

				Psi_0_p = net(X+eps_0)
				Psi_0_n = net(X-eps_0)
				Psi_0   = (Psi_0_p + Psi_0_n)/2
				grad_0  = grad(Psi_0,X,create_graph=True,grad_outputs=torch.ones_like(Psi_0))[0]

				Psi_1_p = net(X+eps_1)
				Psi_1_n = net(X-eps_1)
				Psi_1   = (Psi_1_p + Psi_1_n)/2
				grad_1  = grad(Psi_1,X,create_graph=True,grad_outputs=torch.ones_like(Psi_1))[0]

				r1    = torch.norm(X-R1,dim=1)
				r2    = torch.norm(X-R2,dim=1)
				V     = -1/r1 - 1/r2 #+ 1/R  # is constant offset that does not influence the fitting procedure

				gradloss = torch.sum(0.5*torch.sum(grad_0*grad_1,dim=1)+Psi_0*V*Psi_1)/torch.sum(Psi_0*Psi_1)

				J = gradloss

			elif losses[epoch%len(losses)] == "variance":

				# X = X_all[indx[batch_size*step:(batch_size*(step+1))]]
				# Psi_t = Psi_t_all[indx[batch_size*step:batch_size*(step+1)]]

				#X = (torch.rand(batch_size,3,requires_grad=True)-0.5)*2*5

				X = torch.from_numpy(np.random.normal(0,1,(batch_size,3))*3/2*R.numpy()).type(torch.FloatTensor)
				X.requires_grad = True

				eps_0 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)
				eps_1 = torch.from_numpy(np.random.normal(0,H,X.shape)).type(torch.FloatTensor)

				Psi_0_p = net(X+eps_0)
				Psi_0_n = net(X-eps_0)
				Psi_0   = (Psi_0_p + Psi_0_n)/2

				Psi_1_p = net(X+eps_1)
				Psi_1_n = net(X-eps_1)
				Psi_1   = (Psi_1_p + Psi_1_n)/2

				Lap_0 = eps_0*(grad(Psi_0_p,X,create_graph=True,grad_outputs=torch.ones_like(Psi_0_p))[0]\
				-grad(Psi_0_n,X,create_graph=True,grad_outputs=torch.ones_like(Psi_1_p))[0])/(4*H**2)
				Lap_1 = eps_1*(grad(Psi_1_p,X,create_graph=True,grad_outputs=torch.ones_like(Psi_0_n))[0]\
				-grad(Psi_1_n,X,create_graph=True,grad_outputs=torch.ones_like(Psi_1_n))[0])/(4*H**2)
				#
				Lap_0 = torch.sum(Lap_0,dim=1)
				Lap_1 = torch.sum(Lap_1,dim=1)
				r1    = torch.norm(X-R1,dim=1)
				r2    = torch.norm(X-R2,dim=1)
				V     = -1/r1 - 1/r2 #+ 1/R

				laploss  = torch.mean(((-Lap_0/Psi_0 + V-net.Lambda)*(-Lap_1/Psi_1+ V-net.Lambda)))

				J = laploss

			# w     = Psi_0*Psi_1/Psi_t**2
			# J     = torch.sum(w*(-Lap_0/Psi_0 + (V-net.Lambda))*(-Lap_1/Psi_1+ (V-net.Lambda)))/torch.sum(Psi_0*Psi_1/Psi_t)


			opt.zero_grad()
			J.backward()
			opt.step()


			print("Progress {:2.0%}".format(step /steps), end="\r")



		G=torch.meshgrid([torch.linspace(-7,7,100),torch.linspace(-7,7,100),torch.linspace(-7,7,100)])
		x=G[0].flatten().view(-1,1)
		y=G[1].flatten().view(-1,1)
		z=G[2].flatten().view(-1,1)
		X = torch.cat((x, y, z), 1)
		X.requires_grad=True
		gPsi  = grad(net(X),X,create_graph=True,grad_outputs=torch.ones(len(X)))[0]
		r1    = torch.norm(X-R1,dim=1)
		r2    = torch.norm(X-R2,dim=1)
		V     = -1/r1 - 1/r2 + 1/R
		Psi   = net(X)
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


	plt.axvline(R1.numpy()[0],ls=':',color='k')
	plt.axvline(R2.numpy()[0],ls=':',color='k')

	plt.title("batch_size = "+str(batch_size)+", steps = "+str(steps)+", epochs = "+str(epochs)+", R = "+str(R.item())+", losses = "+str(losses))
	plt.legend(loc="lower center",bbox_to_anchor=[0.5, - 0.4], ncol=8)
	plt.savefig(datetime.datetime.now().strftime("%B%d%Y%I%M%p")+".png")

	return (net,E)
