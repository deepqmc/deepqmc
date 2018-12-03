
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F

from matplotlib import cm
from Combined import *



print(np.sum([1,2,3]))

exit(0)


G=torch.meshgrid([torch.linspace(-1,1,4),torch.linspace(-1,1,4),torch.linspace(-1,1,4)])
x=G[0].flatten().view(-1,1)
y=G[1].flatten().view(-1,1)
z=G[2].flatten().view(-1,1)
r= torch.cat((x, y, z), 1)
t=torch.tensor([-1,1,1]).type(torch.FloatTensor)
r=r.view(2,2,-1,3)
print(r[0]-t*torch.flip(r[1],dims=(0,)))




#print(r[:16]-r[-16:]*t+r[16:32]-r[-32:-16]*t)
#print(r[:32],r[32:])
exit(0)
xmax=2
for alpha in [2,3,5,7]:
    for beta in [8]:
        d = (torch.rand(100,1,requires_grad=False)-0.5)*2*6
        decay=torch.exp(-F.softplus(torch.abs(alpha*torch.norm(d,dim=1))-beta))
        plt.plot(d.numpy(),decay.numpy(),marker='.',ls='',label='a = '+str(alpha)+'  b = '+str(beta))
plt.legend(loc='upper left')
plt.show()
