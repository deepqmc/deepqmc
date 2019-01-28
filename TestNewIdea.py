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
				torch.nn.Linear(100, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 100)
				)

	def forward(self,x):
		return self.NN(x)

net=Net()

X = torch.rand(100).view(1,-1)
Y = net(X).flatten()
Z = 
print(Y)
plt.hist(Y.detach().numpy())
plt.show()
print(X)
