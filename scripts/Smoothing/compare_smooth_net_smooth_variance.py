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

    def forward(self,x,n=10):
        return self.NN(x)*torch.exp(-F.softplus(torch.abs(self.alpha*x)-self.beta))


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


LR=1e-3
BATCH_SIZE=128
H = 0.1
v0 = 2
L = 2
steps=10000

smooth_net=False




plotlist=[] # initialise plotlist
#plotlist.append(make_plot(net_all)) #append untrained network

#net = copy.deepcopy(net_all) # in case I want to train with same initial network


for a in range(10):
    net = Net()
    params = [p for p in net.parameters()]
    #del params[0]
    opt = torch.optim.Adam(params, lr=LR)
    for epochs in range(1):
        start = time.time()
        if smooth_net:
            for step in range(steps):
                print("Progress {:2.0%}".format(step /steps), end="\r")

                X     = ((torch.rand(BATCH_SIZE,1,requires_grad=True))-0.5)*4*L
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
        else:
            for step in range(steps):
                print("Progress {:2.0%}".format(step /steps), end="\r")

                X_0=(torch.rand(BATCH_SIZE,1)-0.5)*4*L
                eps_0=torch.randn(BATCH_SIZE,1)*H
                X_0_p=X_0+eps_0
                X_0_n=X_0-eps_0
                V_0_p=Variable(v0*((X_0_p<=-L/2)+(X_0_p>=L/2)).type(torch.FloatTensor))
                V_0_n=Variable(v0*((X_0_n<=-L/2)+(X_0_n>=L/2)).type(torch.FloatTensor))
                X_0_p=Variable(X_0_p,requires_grad=True)
                X_0_n=Variable(X_0_n,requires_grad=True)
                eps_0=Variable(eps_0)
                Psi_0_p=net(X_0_p)
                Psi_0_n=net(X_0_n)
                Psi_0_p_grad=grad(torch.sum(Psi_0_p),X_0_p,create_graph=True)[0]
                Psi_0_n_grad=grad(torch.sum(Psi_0_n),X_0_n,create_graph=True)[0]
                loss_0=-0.25/H/H*torch.sum(eps_0*(Psi_0_p_grad-Psi_0_n_grad),1,True)\
                +0.5*((V_0_p-net.Lambda)*Psi_0_p+(V_0_n-net.Lambda)*Psi_0_n)

                #X_1=(torch.rand(BATCH_SIZE,1)-0.5)*2*L
                eps_1=torch.randn(BATCH_SIZE,1)*H
                X_1_p=X_0+eps_1
                X_1_n=X_0-eps_1
                V_1_p=Variable(v0*((X_1_p<=-L/2)+(X_1_p>=L/2)).type(torch.FloatTensor))
                V_1_n=Variable(v0*((X_1_n<=-L/2)+(X_1_n>=L/2)).type(torch.FloatTensor))
                X_1_p=Variable(X_1_p,requires_grad=True)
                X_1_n=Variable(X_1_n,requires_grad=True)
                eps_1=Variable(eps_1)
                Psi_1_p=net(X_1_p)
                Psi_1_n=net(X_1_n)
                Psi_1_p_grad=grad(torch.sum(Psi_1_p),X_1_p,create_graph=True)[0]
                Psi_1_n_grad=grad(torch.sum(Psi_1_n),X_1_n,create_graph=True)[0]
                loss_1=-0.25/H/H*torch.sum(eps_1*(Psi_1_p_grad-Psi_1_n_grad),1,True)\
                +0.5*((V_1_p-net.Lambda)*Psi_1_p+(V_1_n-net.Lambda)*Psi_1_n)

                Psi_0=net(X_0)
                loss_2=Psi_0*Psi_0

                loss=torch.mean(loss_0*loss_1/loss_2)
                opt.zero_grad()
                loss.backward()
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
    #if not (i+1)==len(plotlist):
        #plt.plot(x_plot,Psi,label="Episode: "+str(i),ls=':',linewidth=1.,color='grey')
    plt.plot(x_plot,Psi,ls=':',linewidth=1.,color='grey')
    #else:
    #    plt.plot(x_plot,Psi,label="Episode: "+str(i))
Psi_mean=np.mean(np.array(plotlist),axis=0)
Psi_std=np.std(np.array(plotlist),axis=0)
plt.plot(x_plot,analytical_gs(x_plot),label='True WF',color='k')
plt.axvline(-L/2,ls='--',color='k',linewidth=0.5)
plt.axvline(L/2,ls='--',color='k',linewidth=0.5)
plt.plot(x_plot,Psi_mean,label="Mean",color="r")
plt.plot(x_plot,Psi_mean+Psi_std,color="r",ls=':')
plt.plot(x_plot,Psi_mean-Psi_std,color="r",ls=':')
plt.legend(loc='upper right')
plt.show()
