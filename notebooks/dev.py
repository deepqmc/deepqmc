
# coding: utf-8

# In[ ]:


from dlqmc.NN import Net, WaveNet
from dlqmc.Sampler import HMC, HMC_ad, metropolis
from dlqmc.Examples import fit, Potential, Laplacian, Gradient, Gridgenerator


# In[ ]:


#H2+     Energy = -0.6023424   for R = 1.9972
fit(batch_size=10000, n_el=1, steps=500, epochs=1, RR=[[-1, 0, 0], [1., 0, 0]])
#H2		 Energy = -1.173427    for R = 1.40
#fit(batch_size=10000,n_el=2,steps=100,epochs=5,RR=torch.tensor([[-0.7,0,0],[0.7,0,0]]))
#He+	 Energy = -1.9998
#fit(batch_size=10000,n_el=1,steps=100,epochs=5,RR=torch.tensor([[0.,0,0]]),RR_charges=[2])
#He		 Energy = −2.90338583
#fit(batch_size=10000,n_el=2,steps=300,epochs=5,RR=torch.tensor([[0.3,0,0]]),RR_charges=[2])

