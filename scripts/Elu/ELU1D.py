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
            self.Lambda=nn.Parameter(torch.Tensor([-10]))    #eigenvalue
            self.alpha=nn.Parameter(torch.Tensor([4]))#coefficient for decay
            self.beta=nn.Parameter(torch.Tensor([8]))#coefficient for decay
        def forward(self,x):
            #return self.NN(x)[:,0]
            return (self.NN(x)[:,0])*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))

    X_plot = torch.linspace(-6,6,100).view(-1,1)

    LR=1e-3
    V0=2
    c=2
    h=0.1
    net = Net()
    params = [p for p in net.parameters()]
    del params[0]

    opt = torch.optim.Adam(params, lr=LR)


    for epoch in range(epochs):
        X_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*2).type(torch.FloatTensor)
        index = torch.randperm(steps*batch_size)
        X_all.requires_grad = True
        for step in range(steps):
            if False:

                X = X_all[index[step*batch_size:(step+1)*batch_size]]

                Psi=net(X).flatten()
                dx =torch.autograd.grad(Psi,X,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
                ddx=torch.autograd.grad(dx[0].flatten(),X,retain_graph=True,grad_outputs=torch.ones(batch_size))[0].flatten()
                V = (V0*(X>c)+V0*(X<-c)).type(torch.FloatTensor).flatten()
                laploss = torch.mean(Psi*(-0.5*ddx + V*Psi))/torch.mean(Psi**2)
                #laploss = torch.mean(torch.min((-0.5*ddx/torch.max(Psi,torch.ones_like(Psi)*0.01) + V - net.Lambda)**2,torch.ones_like(Psi)*1000))
                J = laploss + 10*(torch.sum(Psi**2)-1)**2


            else:
                X_0 = X_all[index[step*batch_size:(step+1)*batch_size]]
                eps_0=torch.randn(batch_size,1)*h
                X_0_p=X_0+eps_0
                X_0_n=X_0-eps_0
                V_0_p=V = (V0*(X_0_p>c)+V0*(X_0_p<-c)).type(torch.FloatTensor).flatten()
                V_0_n=V = (V0*(X_0_n>c)+V0*(X_0_n<-c)).type(torch.FloatTensor).flatten()
                X_0_p=Variable(X_0_p,requires_grad=True)
                X_0_n=Variable(X_0_n,requires_grad=True)
                eps_0=Variable(eps_0)
                Psi_0_p=net(X_0_p)
                Psi_0_n=net(X_0_n)
                Psi_0_p_grad=grad(torch.sum(Psi_0_p),X_0_p,create_graph=True)[0]
                Psi_0_n_grad=grad(torch.sum(Psi_0_n),X_0_n,create_graph=True)[0]
                loss_0=-0.25/h/h*torch.sum(eps_0*(Psi_0_p_grad-Psi_0_n_grad),1,True)/batch_size\
                +0.5*torch.mean((V_0_p-net.Lambda)*Psi_0_p+(V_0_n-net.Lambda)*Psi_0_n)
                              
                eps_1=torch.randn(batch_size,1)*h
                X_1_p=X_0+eps_1
                X_1_n=X_0-eps_1
                V_1_p=(V0*(X_1_p>c)+V0*(X_1_p<-c)).type(torch.FloatTensor).flatten()
                V_1_n=(V0*(X_1_p>c)+V0*(X_1_p<-c)).type(torch.FloatTensor).flatten()
                X_1_p=Variable(X_1_p,requires_grad=True)
                X_1_n=Variable(X_1_n,requires_grad=True)
                eps_1=Variable(eps_1)
                Psi_1_p=net(X_1_p)
                Psi_1_n=net(X_1_n)
                Psi_1_p_grad=grad(torch.sum(Psi_1_p),X_1_p,create_graph=True)[0]
                Psi_1_n_grad=grad(torch.sum(Psi_1_n),X_1_n,create_graph=True)[0]
                loss_1=-0.25/h/h*torch.sum(eps_1*(Psi_1_p_grad-Psi_1_n_grad),1,True)/batch_size\
                +0.5*torch.mean((V_1_p-net.Lambda)*Psi_1_p+(V_1_n-net.Lambda)*Psi_1_n)
                

                Psi_2=net(X_0)
                loss_2=Psi_2*Psi_2

                J=torch.mean(loss_0*loss_1/(0.*loss_2+1*torch.mean(loss_2)))
                
                X = X_0
                Psi = Psi_2
                V = (V0*(X>c)+V0*(X<-c)).type(torch.FloatTensor).flatten()
                dx =torch.autograd.grad(Psi_2,X_0,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
                ddx=torch.autograd.grad(dx[0].flatten(),X_0,retain_graph=True,grad_outputs=torch.ones(batch_size))[0].flatten()
                
            opt.zero_grad()
            J.backward()
            opt.step()


            print("Progress {:2.0%}".format(step /steps), end="\r")

        X_p = X.flatten().detach().numpy()
        order = np.argsort(X_p)
        plt.plot(X_p[order],Psi.detach().numpy()[order],marker='',label="Psi")
        plt.plot(X_p[order],1/2*ddx.detach().numpy()[order],marker='',label="ddx")
        plt.plot(X_p[order],1/20*(Psi*(-0.5*ddx + V*Psi)/torch.mean(Psi**2)).detach().numpy()[order],marker="",label="E")
        plt.plot(X_p[order],1/10*(0.5*torch.sign(Psi)*ddx/torch.max(torch.abs(Psi),torch.ones_like(Psi)*0.01)) .detach().numpy()[order],marker='',label="ddx/Psi")
        plt.plot(X_p[order],1/50*(V - net.Lambda).detach().numpy()[order],marker='',label="V-lambda")
        plt.plot(X_p[order],1/500*(torch.min((-0.5*torch.sign(Psi)*ddx/torch.max(torch.abs(Psi),torch.ones_like(Psi)*0.01) + V - net.Lambda)**2,torch.ones_like(Psi)*5000)).flatten().detach().numpy()[order],marker='',label="(e_loc-lambda)**2")
        plt.axhline(0,color='k',ls=":")
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
fit(batch_size=5000,steps=10,epochs=8,R1=1,R2=-1)
plt.show()
