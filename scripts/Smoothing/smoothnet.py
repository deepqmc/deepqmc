import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import copy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.NN=nn.Sequential(
                torch.nn.Linear(1, 16),
                # torch.nn.ReLU(),
                # torch.nn.Linear(64, 64),
                # torch.nn.ReLU(),
                # torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1)
                )
        self.Lambda=nn.Parameter(torch.Tensor([0]))#eigenvalue
        self.alpha=nn.Parameter(torch.Tensor([1.]))#coefficient for decay
        self.beta=nn.Parameter(torch.Tensor([1.]))#coefficient for decay

    def forward(self,x):
        return self.NN(x)*((x>-4)*(x<4)).type(torch.FloatTensor)
        #return self.NN(x)*torch.exp(-F.softplus(torch.abs(self.alpha*x)-self.beta))


def analytical_gs(x):
    k=1.0299
    alpha=1.7145
    G=2.8598
    B=1
    return G*np.exp(alpha*x)*(x<=-L/2)+B*np.cos(k*x)*(x>-L/2)*(x<L/2)+G*np.exp(-alpha*x)*(x>=L/2)

def save_psi(net,x_range=(-5,5,1000)):
    x    = Variable(torch.linspace(*x_range)).view(-1,1)
    #eps  = torch.from_numpy(np.random.normal(0,H,len(x))).type(torch.FloatTensor).view(-1,1)
    #Psi  = (net(x+eps)+net(x-eps))/2
    Psi  = net(x)
    x    = x.data.numpy()
    Psi  = Psi.data.numpy()
    Psi /= np.max(np.abs(Psi))
    if np.mean(Psi)<0:
        Psi *= -1
    return Psi


LR=3e-3
BATCH_SIZE=256#128
H = 0.1
v0 = 2
L = 2
steps = 1500



net_all=Net()
plotlist=[] # initialise plotlist
#plotlist.append(make_plot(net_all)) #append untrained network
#net = copy.deepcopy(net_all)	 # in case I want to train with same initial network
nm=1
#Hlist=[0.1,0.05,0.01,0.005]
Hlist=[0.05]
for H in Hlist:
    net = copy.deepcopy(net_all)
    params = [p for p in net.parameters()]
    #del params[0]
    opt = torch.optim.Adam(params, lr=LR)
    for epochs in range(5):
        start = time.time()
        for step in range(steps):
            print("Progress {:2.0%}".format(step /steps), end="\r")

            X     = ((torch.rand(BATCH_SIZE,1,requires_grad=True))-0.5)*4*L
            X[0] = torch.tensor([-4])
            X[-1] = torch.tensor([4])

            X     = X.repeat(1,nm).view(-1,1)
            eps_0 = torch.from_numpy(np.random.normal(0,H,len(X))).type(torch.FloatTensor).view(-1,1)
            eps_1 = torch.from_numpy(np.random.normal(0,H,len(X))).type(torch.FloatTensor).view(-1,1)

            Psi_0_p = net(X+eps_0)
            Psi_0_n = net(X-eps_0)
            Psi_0   = (Psi_0_p + Psi_0_n)/2

            Psi_1_p = net(X+eps_1)
            Psi_1_n = net(X-eps_1)
            Psi_1   = (Psi_1_p + Psi_1_n)/2

            Lap_0 = eps_0*(grad(Psi_0_p,X,create_graph=True,grad_outputs=torch.ones_like(X))[0]-grad(Psi_0_n,X,create_graph=True,grad_outputs=torch.ones_like(X))[0])/(4*H**2)
            Lap_1 = eps_1*(grad(Psi_1_p,X,create_graph=True,grad_outputs=torch.ones_like(X))[0]-grad(Psi_1_n,X,create_graph=True,grad_outputs=torch.ones_like(X))[0])/(4*H**2)


            V     = Variable(v0*((X<=-L/2)+(X>=L/2)).type(torch.FloatTensor))

            J     = torch.mean((-Lap_0+ (V-net.Lambda)*Psi_0)*(-Lap_1+ (V-net.Lambda)*Psi_1)/(Psi_0*Psi_1))

            opt.zero_grad()
            J.backward()
            opt.step()

        print('_____________________________________')
        print('It took', time.time()-start, 'seconds.')
        print('Lambda = '+str(net.Lambda[0].item()))
        print('Alpha  = '+str(net.alpha[0].item()))
        print('Beta   = '+str(net.beta[0].item()))


        plotlist.append(save_psi(net))


plt.figure(figsize=(12,8))
x_plot=np.linspace(-5,5,1000)
for i,Psi in enumerate(plotlist):
    if not (i+1)==len(plotlist):
        plt.plot(x_plot,Psi,label="Episode: "+str(i),ls=':',linewidth=1.)#,color='grey')
    #plt.plot(x_plot,Psi,ls='-',linewidth=2.,label=Hlist[i])
    else:
        plt.plot(x_plot,Psi,label="Episode: "+str(i))
#Psi_mean=np.mean(np.array(plotlist),axis=0)
#Psi_std=np.std(np.array(plotlist),axis=0)
plt.plot(x_plot,analytical_gs(x_plot),label='True WF',color='k')
plt.axvline(-L/2,ls='--',color='k',linewidth=0.5)
plt.axvline(L/2,ls='--',color='k',linewidth=0.5)
#plt.plot(x_plot,Psi_mean,label="Mean",color="r")
#plt.plot(x_plot,Psi_mean+Psi_std,color="r",ls=':')
#plt.plot(x_plot,Psi_mean-Psi_std,color="r",ls=':')

plt.legend(loc='upper right')
plt.show()
