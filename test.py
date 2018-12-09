
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F

from matplotlib import cm
from Combined import *


xmax=2
for alpha in [2,3,5,7]:
    for beta in [8]:
        d = (torch.rand(100,1,requires_grad=False)-0.5)*2*6
        decay=torch.exp(-F.softplus(torch.abs(alpha*torch.norm(d,dim=1))-beta))
        plt.plot(d.numpy(),decay.numpy(),marker='.',ls='',label='a = '+str(alpha)+'  b = '+str(beta))
plt.legend(loc='upper left')
plt.show()

