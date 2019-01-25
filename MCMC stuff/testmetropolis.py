import numpy as np
import matplotlib.pyplot as plt

def distribution(x):
    return (-(x-1)**2+9)/35*(np.sin(x*3)+1)

samples = []
steps = 10000
presteps = 50
maxstepsize = 0.5
for epoch in range(1):
    walker = np.random.uniform(-2,4)
    for i in range(steps+presteps):
        if i > presteps:
            samples.append(walker)

        proposal = walker + np.random.uniform(-maxstepsize,maxstepsize)
        if distribution(proposal)/distribution(walker) > np.random.uniform(0,1):
            walker = proposal



X=np.linspace(-2,4,100)
plt.hist(samples,density=True,bins=70)
plt.plot(X,distribution(X))
plt.show()
