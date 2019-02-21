import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from NN import *
import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import copy

cmap=plt.get_cmap("plasma")

ran = (-4,4)

sigma  = np.array([[3,1],[1,1]])
mu    = np.array([0,0])

gausN = lambda x,mu,sigma : np.linalg.det(sigma)**(-1/2)/(2*np.pi)**(x.shape[-1])*np.exp(-1/2*np.einsum('qj,jk,qk->q',(x-mu),np.linalg.inv(sigma),(x-mu)))


def myhist(X,min=-2,max=2,bins=30):
	p = lambda x,a: torch.exp(-1/2*torch.norm(x[([slice(None,None,1)]+[None for i in range(x.shape[-1])])]-a[None],dim=-1)**2/0.01)/np.sqrt(2*np.pi)
	B=torch.from_numpy(np.array(np.meshgrid(*[np.linspace(min,max,bins) for i in range(X.shape[-1])])).swapaxes(0,-1)).type(torch.FloatTensor)
	res = torch.sum(p(X,B),dim=0)
	return res/torch.sum(res)


def myhist2(X,sigma=0.02):
	f = lambda x: torch.exp(-1/2*torch.norm(x[None,:]-x[:,None],dim=-1)**2/sigma)/np.sqrt(2*np.pi)
	res = torch.sum(f(X),dim=1)
	return res/torch.sum(res)


class Samplenet(nn.Module):
	def __init__(self):
		super(Samplenet, self).__init__()
		self.NN=nn.Sequential(
				torch.nn.Linear(4000, 500),
				torch.nn.ReLU(),
				torch.nn.Linear(500, 500),
				torch.nn.ReLU(),
				torch.nn.Linear(500, 500),
				torch.nn.ReLU(),
				torch.nn.Linear(500, 4000)
				)

	def forward(self,x):
		return self.NN(x)

def fit_sampler(density,steps,dim,samples,ran=[-10,10],LR=0.002):
	net  = Samplenet()
	params = [p for p in net.parameters()]
	opt = torch.optim.Adam(params, lr=LR)

	for i in range(steps):

		print("Progress {:2.0%}".format(i /steps), end="\r")
		X = (torch.rand(dim*samples)).view(1,-1)
		Y = net(X).flatten().reshape(samples,dim)*10
		Z = torch.from_numpy(density(Y.detach().numpy()).flatten()).type(torch.FloatTensor)
		Z = Z / torch.sum(Z)
		Ya = myhist2(Y.flip(dims=(1,)))
		Ya = Ya / torch.sum(Ya)
		ll = torch.sum((Y[:,0]>ran[1]).type(torch.FloatTensor)*(Y[:,0]-ran[1])**2)+torch.sum((Y[:,1]>ran[1]).type(torch.FloatTensor)*(Y[:,1]-ran[1])**2)
		ls = torch.sum((Y[:,0]<ran[0]).type(torch.FloatTensor)*(Y[:,0]-ran[0])**2)+torch.sum((Y[:,1]<ran[0]).type(torch.FloatTensor)*(Y[:,1]-ran[0])**2)
		J = torch.sum((Ya-Z)**2)+ll+ls#-torch.mean(torch.norm(Y[None,:]-Y[:,None],dim=-1))
		opt.zero_grad()
		J.backward()
		opt.step()

	#plt.hist2d(Y[:,0].detach().numpy(),-Y[:,1].detach().numpy(),bins=50,cmap=cmap)
	#plt.show()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(Y[:,0].detach().numpy(),Y[:,1].detach().numpy() ,Z.detach().numpy() )
	ax.scatter(Y[:,0].detach().numpy(),Y[:,1].detach().numpy() ,Ya.detach().numpy() )
	plt.show()

	return net

SAMPLES = 2000
DIM = 2
test_dens = lambda x: gausN(x,mu,sigma)
SNet = fit_sampler(test_dens,40,DIM,SAMPLES)
OUT = []
for i in range(1):
	OUT.append((SNet(torch.rand(SAMPLES*DIM).view(1,-1)*10)).flatten().reshape(SAMPLES,DIM).detach().numpy())
out = np.array(OUT).reshape(-1,2)

x,y = torch.meshgrid([torch.linspace(ran[0],ran[1],100),torch.linspace(ran[0],ran[1],100)])
G=torch.cat((x,y)).view(2,100*100).transpose(0,-1)
P=np.zeros((100*100))
P = gausN(G.detach().numpy(),mu,sigma)
P = P.reshape((100,100))

plt.figure(figsize=(10,5))
plt.subplot2grid((1,2),(0,0))
plt.imshow(P,extent=[ran[0],ran[1],ran[0],ran[1]],cmap=cmap)
plt.plot(out[:,1],-out[:,0],ls='',marker='+',color='k')
plt.subplot2grid((1,2),(0,1))
plt.hist2d(out[:,1],-out[:,0],bins=50,cmap=cmap)#,range=[[-4,4],[-4,4]])
plt.show()


#________________________________________________________________________________
#________________________________________________________________________________
#________________________________________________________________________________
#________________________________________________________________________________
#________________________________________________________________________________
exit(0)




LR=0.001
LR2=0.0001
net  = Samplenet()#
net2 = WaveNet([2,64,64,64,1])
R1 = torch.tensor([-1,0]).type(torch.FloatTensor)
R2 = torch.tensor([1,0]).type(torch.FloatTensor)
R  = torch.tensor([[-1,0],[1,0.]])
params = [p for p in net.parameters()]
opt = torch.optim.Adam(params, lr=LR)

params2 = [p for p in net2.parameters()]
opt2 = torch.optim.Adam(params2, lr=LR2)

epochs = 1
steps  = 100
steps2 = 50
batch_size = 5000
ran = (-4,4)

f = lambda x: net2(x,R)**2

x,y = torch.meshgrid([torch.linspace(ran[0],ran[1],100),torch.linspace(ran[0],ran[1],100)])
G=torch.cat((x,y)).view(2,100,100).transpose(0,-1)
P=np.zeros((100,100))
for i in range(100):

	P[i] = f(G[i]).detach().numpy().flatten()

j=0 #delete later just for plots

#plt.figure(figsize=(15,3))
for epoch in range(epochs):

	start = time.time()



	if epoch==0:
		X_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,2))*3).type(torch.FloatTensor)


	else:
		X_all = torch.zeros(size=(steps*batch_size,2))
		for i in range(steps*batch_size//1000):
			X_i = net(torch.rand(2000).view(1,-1)).detach().flatten().reshape(1000,2)
			X_all[i*1000:(i+1)*1000] = X_i

	index = torch.randperm(steps*batch_size)
	X_all.requires_grad = True

	for step in range(steps):


		X = X_all[index[step*batch_size:(step+1)*batch_size]]

		r1    = torch.norm(X-R1,dim=1)
		r2    = torch.norm(X-R2,dim=1)

		V     = -1/r1 -1/r2

		Psi=net2(X,R).flatten()

		g = torch.autograd.grad(Psi,X,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
		gradloss  = torch.sum(0.5*(torch.sum(g**2,dim=1)) + Psi**2*V)/torch.sum(Psi**2)
		J = gradloss + (torch.sum(Psi**2)-1)**2


		opt2.zero_grad()
		J.backward()
		opt2.step()


		print("Progress {:2.0%}".format(step /steps), end="\r")
	print("\n")

	x,y = torch.meshgrid([torch.linspace(ran[0],ran[1],100),torch.linspace(ran[0],ran[1],100)])
	G=torch.cat((x,y)).view(2,100,100).transpose(0,-1)
	P=np.zeros((100,100))
	for i in range(100):

		P[i] = f(G[i]).detach().numpy().flatten()

	P=P/np.sum(P)

	#plt.subplot2grid((epochs,4),(epoch,0))
	#plt.imshow(P,extent=[ran[0],ran[1],ran[0],ran[1]],cmap=cmap)
	#plt.title("Ground truth")

	Z = torch.from_numpy(P).type(torch.FloatTensor)



	for i in range(steps2):

		print("Progress {:2.0%}".format(i /steps2), end="\r")
		X = torch.rand(2000).view(1,-1)*10
		#Dx = torch.mean(torch.norm(X.reshape(1000,2)[None,:]-X.reshape(1000,2)[:,None],dim=-1))
		Y = net(X).flatten().reshape(1000,2)
		#Dy = torch.mean(torch.norm(Y[None,:]-Y[:,None],dim=-1))
		Z = (net2(Y,R)**2).flatten()
		Z = Z / torch.sum(Z)
		Ya = myhist2(Y.flip(dims=(1,)))
		ll = torch.sum((Y[:,0]>ran[1]).type(torch.FloatTensor)*(Y[:,0]-ran[1])**2)+torch.sum((Y[:,1]>ran[1]).type(torch.FloatTensor)*(Y[:,1]-ran[1])**2)
		ls = torch.sum((Y[:,0]<ran[0]).type(torch.FloatTensor)*(Y[:,0]-ran[0])**2)+torch.sum((Y[:,1]<ran[0]).type(torch.FloatTensor)*(Y[:,1]-ran[0])**2)
		#print(Dy)
		#print(torch.abs(Dx-Dy))
		J = torch.sum((Ya-Z)**2)+ll+ls#+(Dy-Dx)**2
		opt.zero_grad()
		J.backward(retain_graph=True)
		opt.step()


		if (i+1)%(steps2//3)==0 and i!=0:
			#plt.subplot2grid((epochs,4),(j//3,(j%3)+1))
			#plt.hist2d(Y[:,0].detach().numpy(),Y[:,1].detach().numpy(),bins=50,range=np.array([[-4,4],[-4,4]]))
			#plt.title("Sampling, iterations = "+str(i+1))
			j+=1



		#	ax1.plot(np.linspace(ran[0],ran[1],100),Ya.detach().numpy(),label=str(i+1),ls=':')
	#ax1.legend()
	#ax2.imshow(Ya.detach().numpy(),extent=[ran[0],ran[1],ran[0],ran[1]],cmap=cmap)
	#ax2.hist(Y.detach().numpy(),bins=100,density=True)
	#plt.setp(ax1.get_xticklabels(), fontsize=6)


	print('___________________________________________')
	print('It took', time.time()-start, 'seconds.')
	print('\n')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y[:,0].detach().numpy(),Y[:,1].detach().numpy() ,Z.detach().numpy() )
ax.scatter(Y[:,0].detach().numpy(),Y[:,1].detach().numpy() ,Ya.detach().numpy() )
plt.show()


#X_plot = torch.linspace(-5,5,100)
#Y_plot = net2(X_plot.view(-1,1))**2
#plt.plot(X_plot.detach().numpy(),Y_plot.detach().numpy())
#plt.hist(Y.detach().numpy(),bins=100,density=True)
