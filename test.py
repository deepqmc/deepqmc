
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F

print(10**3//2)
exit(0)
G=torch.meshgrid([torch.linspace(-5,5,50),torch.linspace(-5,5,50),torch.linspace(-5,5,50)])
x=G[0].flatten().view(-1,1)
y=G[1].flatten().view(-1,1)
z=G[2].flatten().view(-1,1)
r= torch.cat((x, y, z), 1)
print(r)
exit(0)
xmax=2
for alpha in [2,3,5,7]:
    for beta in [8]:
        d = (torch.rand(100,1,requires_grad=False)-0.5)*2*6
        decay=torch.exp(-F.softplus(torch.abs(alpha*torch.norm(d,dim=1))-beta))
        plt.plot(d.numpy(),decay.numpy(),marker='.',ls='',label='a = '+str(alpha)+'  b = '+str(beta))
plt.legend(loc='upper left')
plt.show()
