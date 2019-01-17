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
import scipy as sp

from pynverse import inversefunc

from scipy.stats import norm






X=np.random.uniform(-5,5,500)



#plt.plot(X,norm.cdf(X,0,0.2),ls='',marker='.')	

def cdf(x):
	return 1/2*sp.special.erfinv(x/np.sqrt(2))
	
def cdf2(x):
	return 1/2*inversefunc(sp.special.erf(x/np.sqrt(2)))
	
#tmp = lambda x: sp.special.erf(x/np.sqrt(2))
tmp = lambda x: x**2
#print(inversefunc(tmp)(X))

#cdf2 = lambda x : 1/2*inversefunc(tmp)(x)
cdf2 = lambda x : inversefunc(tmp)(x)
	
#plt.plot(X,cdf(X),ls='',marker='.')	
#plt.plot(X,X**2,ls='',marker='.')	
plt.plot(X,cdf2(X),ls='',marker='.')	
#print(np.logical_not(np.isinf(cdf(X))))
#plt.hist(cdf(X)[np.logical_not(np.isinf(cdf(X)))],bins=20)
#print(norm.cdf(X,0,0.5))

#plt.hist(norm.cdf(X,0,0.5))
#plt.hist(norm.rvs(size=1000))


#plt.hist(X,bins=20)
#plt.hist(np.sqrt(X*(X>0))-np.sqrt(-X*(X<0)),bins=20)


def ex(x,mu,si):
	return 1/np.sqrt(2*np.pi*si**2)*np.exp(-x**2/(2*si**2))

def exinv(x,mu,si):
	A = -np.sqrt(-2*si**2*np.log(-np.sqrt(2*np.pi*si**2)*x))
	#A[(-2*si**2*np.log(np.sqrt(2*np.pi*si**2)*x)<0)] = 0
	A[(x>=0)]=0
	B = np.sqrt(-2*si**2*np.log(np.sqrt(2*np.pi*si**2)*x))
	#B[(-2*si**2*np.log(np.sqrt(2*np.pi*si**2)*x)<0)] = 0
	B[(x<0)]=0
	#print(2*si**2*np.log(np.sqrt(2*np.pi*si**2)*x))
	#print(A-B)
	#return B*(x>0) - B*(x<0)
	#print(x<0)
	return A + B 

	
#	plt.hist(exinv(X,0,1),bins=10)

plt.show()

exit(0)
def trial_distribution(X1,X2,R):
	return (np.exp(-((X1-R)**2))+(1-np.exp(-((X1-X2)**2))))*np.exp(-((X1-R)**2/50))
	#return np.exp(-np.sum((X-R)**2,axis=1))
	
x_plot=np.linspace(-14,14,100)
R=0	
plt.plot(x_plot,trial_distribution(x_plot,np.array([1]),R))
plt.plot(x_plot,np.exp(-((x_plot-R)**2)))
plt.show()

exit(0)

def metropolis(distribution,startpoint,maxstepsize,steps,presteps=0,interval=None):
	#initialise list to store walker positions
	samples = torch.zeros(steps,len(startpoint))
	#another list for the ratios
	ratios = torch.zeros(steps)
	#initialise the walker at the startposition
	walker = torch.tensor([startpoint]).type(torch.FloatTensor)
	distwalker = distribution(walker)
	#loop over proposal steps
	for i in range(presteps+steps):
		#append position of walker to the sample list in case presteps exceeded
		if i > (presteps-1):
			samples[i-presteps]=(walker)
		#propose new trial position
		#trial = walker + (torch.rand(6)-0.5)*maxstepsize
		pro = torch.zeros(6)
		pro[0:3] = (torch.rand(3)-0.5)*maxstepsize
		trial = walker + pro
		#calculate acceptance propability
		disttrial = distribution(trial)
		#check if in interval
		if not interval is None:
			inint = torch.tensor(all(torch.tensor(interval[0]).type(torch.FloatTensor)<trial[0]) \
			and all(torch.tensor(interval[1]).type(torch.FloatTensor)>trial[0])).type(torch.FloatTensor)
			disttrial = disttrial*inint

		ratio = disttrial/distwalker
		ratios[i-presteps] = ratio
		#accept trial position with respective propability
		if ratio > np.random.uniform(0,1):
			walker = trial
			distwalker = disttrial
	#return list of samples
	print("variance of acc-ratios = " + str((torch.sqrt(torch.mean(ratios**2)-torch.mean(ratios)**2)).data))
	return samples

