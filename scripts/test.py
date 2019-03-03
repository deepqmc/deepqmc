
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import tensorboard
from matplotlib import cm

X = np.random.normal(size=100)

f2 = lambda x: np.exp(-1/2*(x)**2)/np.sqrt(2*np.pi)


res = np.zeros(shape=(len(X)))
for i in range(len(X)):
    #interval = (X[i]+0.1,X[i]-0.1)
    res[i]=np.sum(f(X,X[i]))
    order=np.argsort(X)
    #plt.plot(X[order],f(X,X[i])[order])
print(res)
#plt.plot(X[order],res[order]/40)#/(np.sum(res)))
plt.plot(X[order],res[order]/(np.sum(res)))
plt.plot(X[order],f2(X[order])/np.sum(f2(X[order])),marker='o',ls='')
#plt.show()

def myhist2(X,sigma=0.1):
    f = lambda x,a: np.exp(-1/2*(x-a)**2/sigma)/np.sqrt(2*np.pi)
    res = torch.zeros(size=(len(X),))
    for i in range(len(X)):
        res[i]=torch.sum(f(X,X[i]))
    return res/torch.sum(res)

def myhist(X,min=-0.5,max=0.5,bins=30):
	res = torch.zeros(size=(bins,))
	B=np.linspace(min,max,bins)
	for i in range(bins):
		res[i] = torch.sum(f(X,B[i]))
	return res/torch.sum(res)

#plt.hist(X,density=True)
plt.show()
