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
import pymc3 as pm
import theano.tensor as T

X = np.random.uniform(-3,3,size=(100))

#alpha_=np.ones(1)
#Y = np.einsum("i,ji",alpha_,X)+np.random.normal(size=X.shape[0])
Y = np.exp(-X**2)
plt.plot(X,Y,ls='',marker='o')
#plt.hist(Y)
#plt.show()

#exit(0)
test_model = pm.Model()

with test_model:

	alpha = pm.Normal('alpha',mu=0,tau=0.1,shape=(100))

	tmp = T.dot(X,alpha)
	#tmp = X
	y = pm.Normal('y',mu=tmp , tau =0.01, observed = Y)

with test_model:
	#help(pm.NUTS)
	step = pm.NUTS()
	nuts_trace = pm.sample(draws=30,tune=5,step=step,njobs=1)

pm.traceplot(nuts_trace)
plt.show()
