
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F

from matplotlib import cm
from Combined import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.NN=nn.Sequential(
                torch.nn.Linear(3, 64),
                torch.nn.ELU(),
                torch.nn.Linear(64, 64),
                torch.nn.ELU(),
                torch.nn.Linear(64, 64),
                torch.nn.ELU(),
                torch.nn.Linear(64, 1)
                )
        self.Lambda=nn.Parameter(torch.Tensor([-1]))	#eigenvalue
        self.alpha=nn.Parameter(torch.Tensor([1]))#coefficient for decay
        self.beta=nn.Parameter(torch.Tensor([8]))#coefficient for decay
    def forward(self,x):
        return self.NN(x)
        #return self.NN(d)[:,0]*torch.exp(-F.softplus(torch.abs(self.alpha*torch.norm(x,dim=1))-self.beta))

net=Net()
#x=Variable(torch.tensor([2.,2.,3.]),requires_grad=True)
#y = net(x)
#grad = torch.autograd.grad(y,x, create_graph=True)#,grad_outputs=torch.ones(2))
#lap  = torch.autograd.grad(grad,x,grad_outputs=torch.ones(3))
#print(grad,lap)


#X1=Variable(torch.tensor([1,2,3,4.]),requires_grad=True)
#X2=Variable(2*torch.tensor([1,2,3,4.]),requires_grad=True)
#X3=Variable(3*torch.tensor([1,2,3,4.]),requires_grad=True)
X1=torch.tensor([1.,2])
X1.requires_grad=True
X2=torch.tensor([2.,3])
X2.requires_grad=True
X3=torch.tensor([3.,4])
X3.requires_grad=True
X=torch.cat([X1,X2,X3], dim=0)
X=X.reshape(3,2).transpose(0,1)
#X=torch.stack([X1,X2,X3]).transpose(0,1)

#n=len(X)
#print(X)
#print((X**2).type())
A=net(X).flatten()
#A=X1**2+X2**2+X3**2
print(A)
#G1 =torch.autograd.grad(A,X1,grad_outputs=torch.ones_like(A),create_graph=True,retain_graph=True)

G1 =torch.autograd.grad(A,X1,create_graph=True,retain_graph=True,grad_outputs=torch.ones(2))
print(G1)
ddx=torch.autograd.grad(G1[0].flatten(),X1,retain_graph=True,grad_outputs=torch.ones(2))
print(ddx)
G2 =torch.autograd.grad(A,X2,create_graph=True,retain_graph=True,grad_outputs=torch.ones(2))
print(G1)
ddy=torch.autograd.grad(G2[0].flatten(),X2,retain_graph=True,grad_outputs=torch.ones(2))
print(ddy)
G3 =torch.autograd.grad(A,X3,create_graph=True,retain_graph=True,grad_outputs=torch.ones(2))
print(G3)
ddz=torch.autograd.grad(G3[0].flatten(),X3,retain_graph=True,grad_outputs=torch.ones(2))
print(ddz)

#ddx=torch.autograd.grad(G1,X1,grad_outputs=torch.ones(1,3))
#G2 =torch.autograd.grad(A,X2,grad_outputs=torch.ones(n),create_graph=True,retain_graph=True)
#ddy=torch.autograd.grad(G2,X2,grad_outputs=torch.ones(n))
#G3 =torch.autograd.grad(A,X3,grad_outputs=torch.ones(n),create_graph=True,retain_graph=True)
#ddz=torch.autograd.grad(G3,X3,grad_outputs=torch.ones(n))




exit(0)

#G2=grad(A,X2,create_graph=True,grad_outputs=torch.ones(n))[0]
#G3=grad(A,X3,create_graph=True,grad_outputs=torch.ones(n))[0]

ddx=grad(torch.sum(G1[0]),X1,allow_unused=True)
#ddy=grad(G2,X2,grad_outputs=torch.ones(n), retain_graph=True)
#ddz=grad(G3,X3,grad_outputs=torch.ones(n), retain_graph=True)
print(ddx)
#print(ddx,ddy,ddz)
#print(grad(G1[0][:,1],X,grad_outputs=torch.ones(n)))
exit(0)



L=torch.zeros_like(X)
for i in range(3):
    print(grad(G[0][:,i],X[:,i],grad_outputs=torch.ones(n)))
    L[:,i]=grad(G[0][:,i],X[:,i],grad_outputs=torch.ones(n))[0]
L=torch.sum(L,dim=1)
print(X)
print(A)
print(G)
exit(0)

xmax=2
for alpha in [2,3,5,7]:
    for beta in [8]:
        d = (torch.rand(100,1,requires_grad=False)-0.5)*2*6
        decay=torch.exp(-F.softplus(torch.abs(alpha*torch.norm(d,dim=1))-beta))
        plt.plot(d.numpy(),decay.numpy(),marker='.',ls='',label='a = '+str(alpha)+'  b = '+str(beta))
plt.legend(loc='upper left')
plt.show()
