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


def fit(batch_size=2056,steps=15,epochs=4,R1=1.5,R2=-1.5,losses=["variance","energy","symmetry"]):
    class Net(nn.Module):
    	def __init__(self):
    		super(Net, self).__init__()
    		self.NN=nn.Sequential(
    				torch.nn.Linear(1, 64),
    				torch.nn.ELU(),
    				torch.nn.Linear(64, 64),
    				torch.nn.ELU(),
    				torch.nn.Linear(64, 64),
    				torch.nn.ELU(),
    				#torch.nn.Linear(64, 64),
    				#torch.nn.ELU(),
    				torch.nn.Linear(64, 1)
    				)
    		self.Lambda=nn.Parameter(torch.Tensor([-10]))	#eigenvalue
    		self.alpha=nn.Parameter(torch.Tensor([4]))#coefficient for decay
    		self.beta=nn.Parameter(torch.Tensor([10]))#coefficient for decay
    	def forward(self,x):
    		return (self.NN(x)[:,0])*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))

    X_plot = torch.linspace(-6,6,100).view(-1,1)

    LR=1e-4
    V0=20
    c=2

    net = Net()
    params = [p for p in net.parameters()]
    del params[0]

    opt = torch.optim.Adam(params, lr=LR)


    for epoch in range(epochs):
        X_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*3).type(torch.FloatTensor)
        index = torch.randperm(steps*batch_size)
        X_all.requires_grad = True
        for step in range(steps):

            X = X_all[index[step*batch_size:(step+1)*batch_size]]

            Psi=net(X).flatten()
            dx =torch.autograd.grad(Psi,X,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
            ddx=torch.autograd.grad(dx[0].flatten(),X,retain_graph=True,grad_outputs=torch.ones(batch_size))[0].flatten()
            V = (V0*(X>c)+V0*(X<-c)).type(torch.FloatTensor).flatten()
            laploss = torch.mean(Psi*(-0.5*ddx + V*Psi))/torch.mean(Psi**2)
            #laploss = torch.mean(torch.min((-0.5*ddx/torch.max(Psi,torch.ones_like(Psi)*0.01) + V - net.Lambda)**2,torch.ones_like(Psi)*1000))
            J = laploss #+ 10*(torch.sum(Psi**2)-1)**2

            opt.zero_grad()
            J.backward()
            #for p in params:
            #	print(p.grad)
            opt.step()


            print("Progress {:2.0%}".format(step /steps), end="\r")


        plt.plot(X.flatten().detach().numpy(),Psi.detach().numpy(),ls="",marker='x',label="Psi")
        plt.plot(X.flatten().detach().numpy(),ddx.detach().numpy(),ls="",marker='x',label="ddx")
        plt.plot(X.flatten().detach().numpy(),1/100*(-0.5*ddx/torch.max(Psi,torch.ones_like(Psi)*0.01)).detach().numpy(),ls="",marker='x',label="ddx/Psi")
        plt.plot(X.flatten().detach().numpy(),1/500*(torch.min((-0.5*ddx/torch.max(Psi,torch.ones_like(Psi)*0.01) + V - net.Lambda)**2,torch.ones_like(Psi)*5000)).flatten().detach().numpy(),ls="",marker='x')
        #plt.ylim(-15,15)
        plt.legend()
        plt.show()
        Xe=torch.linspace(-5,5,100).view(-1,1)
        Xe.requires_grad=True
        Psi   = net(Xe)
        gPsi  = grad(Psi,Xe,create_graph=True,grad_outputs=torch.ones(len(Xe)))[0]
        V     = (V0*(Xe>c)+V0*(Xe<-c)).type(torch.FloatTensor)
        E     = (torch.mean(torch.sum(gPsi**2,dim=1)/2+Psi**2*V)/torch.mean(Psi**2)).item()*27.211386 # should give ~ -0.6023424 (-16.4) for hydrogen ion at (R ~ 2 a.u.)


        print('___________________________________________')
        print('Lambda = '+str(net.Lambda[0].item()*27.211386))
        print('Energy = '+str(E))
        print('Alpha  = '+str(net.alpha[0].item()))
        print('Beta   = '+str(net.beta[0].item()))
        print('\n')
        plt.figure()
        Psi_plot = net(X_plot).detach().numpy()

        plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),linewidth =3)

        plt.show()
fit(batch_size=1000,steps=400,epochs=8,R1=1,R2=-1)
plt.show()
