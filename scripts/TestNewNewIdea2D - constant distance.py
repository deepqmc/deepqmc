import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F

cmap=plt.get_cmap("plasma")

ran = (-8,8)

sigma  = np.array([[5,1],[1,1]])
sigma2 = np.array([[1,0],[0,5]])
mu     = np.array([-5,0.5])
mu2    = np.array([2,-1])

gausN = lambda x,mu,sigma : np.linalg.det(sigma)**(-1/2)/(2*np.pi)**(x.shape[-1])*np.exp(-1/2*np.einsum('qj,jk,qk->q',(x-mu),np.linalg.inv(sigma),(x-mu)))



def myhist(X,min=-2,max=2,bins=30):
	p = lambda x,a: torch.exp(-1/2*torch.norm(x[([slice(None,None,1)]+[None for i in range(x.shape[-1])])]-a[None],dim=-1)**2/0.01)/np.sqrt(2*np.pi)
	B=torch.from_numpy(np.array(np.meshgrid(*[np.linspace(min,max,bins) for i in range(X.shape[-1])])).swapaxes(0,-1)).type(torch.FloatTensor)
	res = torch.sum(p(X,B),dim=0)
	return res/torch.sum(res)


def myhist2(X,sigma=0.02):
	f = lambda x: torch.exp(-1/2*torch.norm(x[None,:]-x[:,None],dim=-1)**2/sigma)/np.sqrt(2*np.pi)
	res = torch.mean(f(X),dim=1)
	return res


class Samplenet(nn.Module):
	def __init__(self):
		super(Samplenet, self).__init__()
		self.NN=nn.Sequential(
				torch.nn.Linear(4000, 500),
				torch.nn.ReLU(),
				torch.nn.Linear(500, 500),
				torch.nn.ReLU(),
				#torch.nn.Linear(500, 500),
				#torch.nn.ReLU(),
				torch.nn.Linear(500, 4000)
				)

	def forward(self,x):
		return self.NN(x)

def fit_sampler(density,steps,dim,samples,ran=[-20,20],LR=0.001):
	net  = Samplenet()
	params = [p for p in net.parameters()]
	opt = torch.optim.Adam(params, lr=LR)
	OUT = []
	for i in range(steps):

		print("Progress {:2.0%}".format(i /steps), end="\r")
		X = (torch.rand(dim*samples)).view(1,-1)*10
		Y = net(X).flatten().reshape(samples,dim)
		Z = torch.from_numpy(density(Y.detach().numpy()).flatten()).type(torch.FloatTensor)
		Z = Z / torch.sum(Z)
		Ya = myhist2(Y.flip(dims=(1,)),sigma=0.03)

		ll = torch.sum((Y[:,0]>ran[1]).type(torch.FloatTensor)*(Y[:,0]-ran[1])**2)+torch.sum((Y[:,1]>ran[1]).type(torch.FloatTensor)*(Y[:,1]-ran[1])**2)
		ls = torch.sum((Y[:,0]<ran[0]).type(torch.FloatTensor)*(Y[:,0]-ran[0])**2)+torch.sum((Y[:,1]<ran[0]).type(torch.FloatTensor)*(Y[:,1]-ran[0])**2)
	
		J = torch.sum((Ya/torch.sum(Ya)-Z)**2)+ll+ls#+torch.mean(Ya)**2
		
		
		if i%5==0:
			OUT.append(Y.reshape(samples,dim).detach().numpy())
			
		opt.zero_grad()
		J.backward()
		opt.step()


	#fig = plt.figure()
	#ax = fig.add_subplot(111, projection='3d')
	#ax.scatter(Y[:,0].detach().numpy(),Y[:,1].detach().numpy() ,Z.detach().numpy() )
	#ax.scatter(Y[:,0].detach().numpy(),Y[:,1].detach().numpy() ,Ya.detach().numpy() )
	#plt.show()

	return net,OUT

SAMPLES = 2000
DIM = 2
test_dens = lambda x: gausN(x,mu,sigma)+gausN(x,mu2,sigma2)
SNet,OUT2  = fit_sampler(test_dens,100,DIM,SAMPLES)

for i in range(4):
	for j in range(5):
		plt.subplot2grid((4,5),(i,j))
		plt.hist2d(OUT2[i*5+j][:,1],-OUT2[i*5+j][:,0],bins=50,cmap=cmap)
	

plt.show()

OUT = []
for i in range(1):
	OUT.append((SNet(torch.rand(SAMPLES*DIM).view(1,-1)*100)).flatten().reshape(SAMPLES,DIM).detach().numpy())
out = np.array(OUT).reshape(-1,2)

x,y = torch.meshgrid([torch.linspace(ran[0],ran[1],100),torch.linspace(ran[0],ran[1],100)])
G=torch.cat((x,y)).view(2,100*100).transpose(0,-1)
P=np.zeros((100*100))
P = test_dens(G.detach().numpy())
P = P.reshape((100,100))

plt.figure(figsize=(10,5))
plt.subplot2grid((1,2),(0,0))
plt.imshow(P,extent=[ran[0],ran[1],ran[0],ran[1]],cmap=cmap)
plt.plot(out[:,1],-out[:,0],ls='',marker='+',color='k')
plt.subplot2grid((1,2),(0,1))
plt.hist2d(out[:,1],-out[:,0],bins=50,cmap=cmap)#,range=[[-4,4],[-4,4]])
plt.show()

