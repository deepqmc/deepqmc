import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import copy



f = lambda x,a: torch.exp(-1/2*(x-a)**2/0.01)/np.sqrt(2*np.pi)

#def almost_sigmoid(x,a,x0,x1):
#	return a*(x-x0)/torch.sqrt(a**2*(x-x0)**2+1)-a*(x-x1)/torch.sqrt(a**2*(x-x1)**2+1)

m=20


def myhist(X,min=-0.5,max=0.5,bins=30):
	res = torch.zeros(size=(bins,))
	B=np.linspace(min,max,bins)
	for i in range(bins):
		res[i] = torch.sum(f(X,B[i]))
	return res/torch.sum(res)

def myhist2(X,sigma=0.1):
    f = lambda x,a: torch.exp(-1/2*(x-a)**2/sigma)/np.sqrt(2*np.pi)
    res = torch.zeros(size=(len(X),))
    for i in range(len(X)):
        res[i]=torch.sum(f(X,X[i]))
    return res/torch.sum(res)

class Samplenet(nn.Module):
	def __init__(self):
		super(Samplenet, self).__init__()
		self.NN=nn.Sequential(
				torch.nn.Linear(m, 100),
				torch.nn.ReLU(),
				torch.nn.Linear(100, 100),
				torch.nn.ReLU(),
				torch.nn.Linear(100, 100),
				torch.nn.ReLU(),
				torch.nn.Linear(100, m)
				)

	def forward(self,x):
		return self.NN(x)

class Wavenet(nn.Module):
	def __init__(self):
		super(Wavenet, self).__init__()
		self.NN=nn.Sequential(
				torch.nn.Linear(2, 64),
				torch.nn.ELU(),
				torch.nn.Linear(64, 64),
				torch.nn.ELU(),
				#torch.nn.Linear(64, 64),
				#torch.nn.ELU(),
				torch.nn.Linear(64, 64),
				torch.nn.ELU(),
				torch.nn.Linear(64, 1)
				)
		#self.Lambda=nn.Parameter(torch.Tensor([-1]))	#eigenvalue

	def forward(self,x):
		d = torch.zeros(len(x),2)
		d[:,0] = torch.norm(x-R1,dim=1)
		d[:,1] = torch.norm(x-R2,dim=1)
		r = torch.erf(d/0.5)/d
		return self.NN(r)


LR=0.001
net  = Samplenet()#
net2 = Wavenet()
R1 = torch.tensor([-1]).type(torch.FloatTensor)
R2 = torch.tensor([1]).type(torch.FloatTensor)

params = [p for p in net.parameters()]
opt = torch.optim.Adam(params, lr=LR)

params2 = [p for p in net2.parameters()]
opt2 = torch.optim.Adam(params2, lr=LR)

epochs = 3
steps  = 100
steps2 = 1000
batch_size = 100
ran = (-5,5)
Z = (net2(torch.linspace(ran[0],ran[1],batch_size).view(-1,1)).flatten())**2


for epoch in range(epochs):
	fig=plt.figure()
	ax1 = plt.subplot(311)
	ax2 = plt.subplot(312, sharex = ax1)
	ax3 = plt.subplot(313, sharex = ax1)
	start = time.time()

	if epoch==0:
		X_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3).type(torch.FloatTensor)
	else:
		pass


	index = torch.randperm(steps*batch_size)
	X_all.requires_grad = True

	for step in range(steps):


		X = X_all[index[step*batch_size:(step+1)*batch_size]]

		r1    = torch.norm(X-R1,dim=1)
		r2    = torch.norm(X-R2,dim=1)

		V     = -1/r1 -1/r2

		Psi=net2(X).flatten()

		g = torch.autograd.grad(Psi,X,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
		gradloss  = torch.sum(0.5*(torch.sum(g**2,dim=1)) + Psi**2*V)/torch.sum(Psi**2)
		J = gradloss + (torch.sum(Psi**2)-1)**2


		opt2.zero_grad()
		J.backward()
		opt2.step()


		print("Progress {:2.0%}".format(step /steps), end="\r")
	print("\n")
	X_plot = torch.linspace(ran[0],ran[1],100)
	Y_plot = net2(X_plot.view(-1,1))**2
	ax1.plot(X_plot.detach().numpy(),Y_plot.detach().numpy(),color='k',label='WF')

	Z = (net2(X.view(-1,1)).flatten())**2

	#check if reintializing is better than keeping (would expect keeping is better in higher dimensions)

	net  = Samplenet()
	params = [p for p in net.parameters()]
	opt = torch.optim.Adam(params, lr=LR)
	for i in range(steps2):

		print("Progress {:2.0%}".format(i /steps2), end="\r")
		X = ((torch.rand(m)-1/2)*100).view(1,-1)
		Y = net(X).flatten()
		Z = (net2(Y.view(-1,1)).flatten())**2
		Z = Z/torch.sum(Z)
		Ya = myhist2(Y)
		#print(torch.sum((Y>ran[1]).type(torch.FloatTensor)*(Y-ran[1])**2))
		ll = torch.sum((Y>ran[1]).type(torch.FloatTensor)*(Y-ran[1])**2)
		ls = torch.sum((Y<ran[0]).type(torch.FloatTensor)*(Y-ran[0])**2)
		J = torch.sum((Ya-Z)**2)+ll+ls
		opt.zero_grad()
		J.backward(retain_graph=True)
		opt.step()

		if (i+1)%500==0 and i!=0:
			Y = Y.flatten()
			order=np.argsort(Y.detach())
			ax2.plot(Y[order].detach().numpy(),Ya.detach().numpy()[order],label=str(i+1),ls=':',marker='o')
			ax2.plot(Y[order].detach().numpy(),Z.detach().numpy()[order],ls=':',marker='o')

	X_all = torch.zeros(size=(steps*batch_size,1))
	for i in range(steps*batch_size//m):
		X_i = net(((torch.rand(m)-0.5)*100).view(1,-1)).detach().transpose(0,1)
		X_all[i*m:(i+1)*m] = X_i
	ax3.hist(X_all.detach().numpy(),bins=50)

	ax1.legend()
	#ax3.hist(Y.detach().numpy(),bins=100,density=True)
	plt.setp(ax1.get_xticklabels(), fontsize=6)
	plt.show()

	print('___________________________________________')
	print('It took', time.time()-start, 'seconds.')
	print('\n')


#X_plot = torch.linspace(-5,5,100)
#Y_plot = net2(X_plot.view(-1,1))**2
#plt.plot(X_plot.detach().numpy(),Y_plot.detach().numpy())
#plt.hist(Y.detach().numpy(),bins=100,density=True)

plt.show()

exit(0)

steps = 100#
pics = 5
ran = (-5,5)

Z = (net2(torch.linspace(ran[0],ran[1],100).view(-1,1)).flatten())**2
plt.plot(np.linspace(ran[0],ran[1],100),Z.detach().numpy())
Y=np.zeros(100)
#for i in range(100):
	#B=np.linspace(-5,5,100)
	#Y+=f(torch.linspace(ran[0],ran[1],100),B[i]).detach().numpy()
	#plt.plot(np.linspace(ran[0],ran[1],100),f(torch.linspace(ran[0],ran[1],100),B[i]).detach().numpy())
#plt.plot(np.linspace(ran[0],ran[1],100),Y)
plt.show()

for i in range(steps):

	print("Progress {:2.0%}".format(i /steps), end="\r")

	X = torch.rand(1000).view(1,-1)
	Y = net(X).flatten()
	Ya = myhist(Y,ran[0],ran[1],100)

	J = torch.sum((Ya-Z)**2)

	opt.zero_grad()
	J.backward(retain_graph=True)
	opt.step()

	if (i%(steps//pics))==0:
		print(i)
		plt.subplot2grid((steps//(steps//pics),1),(i//(steps//pics),0))
		plt.plot(np.linspace(ran[0],ran[1],100),Ya.detach().numpy())
		plt.plot(np.linspace(ran[0],ran[1],100),Z.detach().numpy())

plt.show()

plt.hist(Y.detach().numpy(),bins=100,density=True)
plt.show()
