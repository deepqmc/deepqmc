import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.NN=nn.Sequential(
                torch.nn.Linear(1, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
                )
        self.Lambda=nn.Parameter(torch.Tensor([0]))#eigenvalue
        self.alpha=nn.Parameter(torch.Tensor([1.]))#coefficient for decay
        self.beta=nn.Parameter(torch.Tensor([1.]))#coefficient for decay

    def forward(self,x):
        return self.NN(x)*torch.exp(-F.softplus(torch.abs(self.alpha*x)-self.beta))


def laplacian(net,x,h=0.01):
     return (net(x+h)+net(x-h)-2*net(x))/h**2

def make_plot(net):
    x    = Variable(torch.linspace(0.1,10)).view(-1,1)
    Psi  = net(x)#*x
    x    = x.data.numpy()
    Psi  = Psi.data.numpy()
    Psi /= np.max(np.abs(Psi))
    if np.mean(Psi)<0:
        Psi *= -1
    return Psi/x

def analytical_gs(r,a_0=1):   #actually a=0.52*10**(-10))
	return 2*a_0**(-3/2)*np.exp(-r/a_0)
	
LR=1e-3
BATCH_SIZE=512

net = Net()
params = [p for p in net.parameters()]
#del params[0]
opt = torch.optim.Adam(params, lr=LR)

plotlist=[] # initialise plotlist
plotlist.append(make_plot(net)) #append untrained network


def Hamiltonian(net,x,l=0):
    return -laplacian(net,x)+(-2*torch.abs(1/x)+(l*(l+1))/x**2)*net(x)


for epochs in range(5):
    start = time.time()
    for step in range(5000):

        X_0 = (torch.rand(BATCH_SIZE,1,requires_grad=True))*5+0.1 
	#create smples in interval [0.1,5.1)

        loss = torch.mean((Hamiltonian(net,X_0)-net.Lambda*net(X_0))**2/net(X_0)**2)  
	#variance loss

        #loss = torch.mean(net(X_0)*Hamiltonian(net,X_0)/net(X_0)**2) 
	#variational principle

        #Psi=net(X_0)
        #loss = torch.mean(0.5*grad(Psi,X_0,create_graph=True,grad_outputs=torch.ones_like(Psi))[0]**2+net(X_0)*-1/X_0*net(X_0))/torch.mean(net(X_0)**2)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print('e # '+str(epochs+1)+'_____________________________________')
    print('It took', time.time()-start, 'seconds.')
    print('Lambda = '+str(net.Lambda[0].item()))
    print('Alpha  = '+str(net.alpha[0].item()))
    print('Beta   = '+str(net.beta[0].item()))


    plotlist.append(make_plot(net))


plt.figure(figsize=(12,8))
x_plot=np.linspace(0.1,10,100)
for i,Psi in enumerate(plotlist):
    if not (i+1)==len(plotlist):
        plt.plot(x_plot,Psi,label="Episode: "+str(i),ls=':',linewidth=1.)
    else:
        plt.plot(x_plot,Psi,label="Episode: "+str(i))
plt.plot(x_plot,analytical_gs(x_plot),label='True WF')
plt.legend(loc='upper right')
plt.show()
