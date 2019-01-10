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


cmap=plt.get_cmap("plasma")

def metropolis(distribution,interval,startpoint,maxstepsize,steps,presteps=0):
	#initialise list to store walker positions
	samples = torch.zeros(steps,len(startpoint))
	#initialise the walker at the startposition
	walker = torch.tensor([startpoint]).type(torch.FloatTensor)
	distwalker = distribution(walker)
	#loop over proposal steps
	for i in range(presteps+steps):
		#append position of walker to the sample list in case presteps exceeded
		if i > (presteps-1):
			samples[i-presteps]=(walker)
		#propose new trial position
		trial = walker + (torch.rand(3)-0.5)*maxstepsize
		#check if in interval
		inint = torch.tensor(all(torch.tensor(interval[0]).type(torch.FloatTensor)<trial[0]) \
		and all(torch.tensor(interval[1]).type(torch.FloatTensor)>trial[0])).type(torch.FloatTensor)
		#calculate acceptance propability
		disttrial = distribution(trial)*inint
		ratio = disttrial/distwalker
		#accept trial position with respective propability
		if ratio > np.random.uniform(0,1):
			walker = trial
			distwalker = disttrial
	#return list of samples
	return samples

def fit(batch_size,steps,epochs,R1,R2,losses):	# main function 

	class Net(nn.Module):         #define neural network
		def __init__(self):
			super(Net, self).__init__()
			self.NN=nn.Sequential(
					torch.nn.Linear(2, 64),
					torch.nn.ELU(),
					torch.nn.Linear(64, 64),
					torch.nn.ELU(),
					torch.nn.Linear(64, 64),
					torch.nn.ELU(),
					#torch.nn.Linear(64, 64),
					#torch.nn.ELU(),
					torch.nn.Linear(64, 1)
					)
			self.Lambda=nn.Parameter(torch.Tensor([-1]))	#energy eigenvalue
			
		def forward(self,x):                #define forward pass
			d = torch.zeros(len(x),2)       #get distances
			d[:,0] = torch.norm(x-R1,dim=1)
			d[:,1] = torch.norm(x-R2,dim=1)
			r = torch.erf(d/0.01)/d         #get inverse distances
			return self.NN(r)[:,0]

	LR=1e-3   #learning rate

	R1    = torch.tensor([R1,0,0]).type(torch.FloatTensor)  #position of atom 1
	R2    = torch.tensor([R2,0,0]).type(torch.FloatTensor)  #position of atom 2 
	R     = torch.norm(R1-R2)                               #distance between atoms


	net = Net()                                      #instance object of network class
	params = [p for p in net.parameters()]           #get list of network parameters 
	opt = torch.optim.Adam(params, lr=LR)    		 #pass parameters to optimizer (for backpropagation)
	
	E = 100         #initialise energy variable
	E_min = 100     #initialise minimal energy variable
	
	plt.figure()    #create figure for plotting
	

	
	for epoch in range(epochs):    #iterate over epochs

		print("epoch " +str(1+epoch)+" of "+str(epochs)+":")  #some prints to monitor the training
		if losses[epoch%len(losses)] == "energy":
			print("minimize energy")
		elif losses[epoch%len(losses)] == "variance":
			print("minimize variance of energy")
		else:
			print("loss error, check losses:"+str(losses[epoch%len(losses)] ))
			
		start = time.time()      #initial time (for monitoring)


		
		X1_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*(R/2).numpy()).type(torch.FloatTensor)    #create spacial samples for training 
		X2_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*(R/2).numpy()).type(torch.FloatTensor)
		X3_all = torch.from_numpy(np.random.normal(0,1,(batch_size*steps,1))*(R/2).numpy()).type(torch.FloatTensor)
		X_all=torch.cat([X1_all,X2_all,X3_all], dim=0).reshape(3,batch_size*steps).transpose(0,1)
		
		X_all.requires_grad = True                  #needed for the derivation
		X1_all.requires_grad = True
		X2_all.requires_grad = True
		X3_all.requires_grad = True
		
		index = torch.randperm(steps*batch_size)    #get random indices 


		for step in range(steps): #iterate over steps per epoch

			if losses[epoch%len(losses)] == "energy":   #choose energy loss

				X=X_all[index[step*batch_size:(step+1)*batch_size]]    #get subset of the training samples
				
				Psi=net(X).flatten()    #compute network output (wavefunction)
				
				grad_X = grad(Psi,X,create_graph=True,grad_outputs=torch.ones_like(Psi))[0]    #differentiate with respect to input X

				r1    = torch.norm(X-R1,dim=1)    #compute electron nuclei distances
				r2    = torch.norm(X-R2,dim=1)
				V     = -1/r1 - 1/r2              #compute potential energy
 
				gradloss = torch.sum(0.5*torch.sum(grad_X**2,dim=1)+Psi*V*Psi)/torch.sum(Psi**2)   #compute energy (as lossfunction)

				J = gradloss + (torch.mean(Psi**2)-1)**2   #add a loss to keep wavefunction small for increased stability

			elif losses[epoch%len(losses)] == "variance":   #choose variance loss

				X1 = X1_all[index[step*batch_size:(step+1)*batch_size]]   
				X2 = X2_all[index[step*batch_size:(step+1)*batch_size]]
				X3 = X3_all[index[step*batch_size:(step+1)*batch_size]]
				X=torch.cat([X1,X2,X3], dim=0).reshape(3,batch_size).transpose(0,1)    #get subset of the training samples

				Psi=net(X).flatten()  #compute network output (wavefunction)
				
				dx =torch.autograd.grad(Psi,X1,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size)) #partially differentiate with respect to input x
				ddx=torch.autograd.grad(dx[0].flatten(),X1,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]    #second partial derivative with respect to input x
				dy =torch.autograd.grad(Psi,X2,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
				ddy=torch.autograd.grad(dy[0].flatten(),X2,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
				dz =torch.autograd.grad(Psi,X3,create_graph=True,retain_graph=True,grad_outputs=torch.ones(batch_size))
				ddz=torch.autograd.grad(dz[0].flatten(),X3,retain_graph=True,grad_outputs=torch.ones(batch_size))[0]
				
				lap_X = (ddx+ddy+ddz).flatten()   #compute laplacian from partial derivatives

				r1    = torch.norm(X-R1,dim=1)   #compute electron nuclei distances
				r2    = torch.norm(X-R2,dim=1)
				V     = -1/r1 - 1/r2 			 #compute potential energy

				laploss = torch.mean((-0.5*lap_X/Psi + V - net.Lambda)**2) #compute variance of energy (as lossfunction)
				
				J = laploss + (torch.mean(Psi**2)-1)**2   #add a loss to keep wavefunction small for increased stability
				

				#plt.subplot2grid((1,2),(0,0))
				#plt.plot(X1.detach().numpy(),(-0.5*lap_X/Psi).detach().numpy(),ls='',marker='.')
				#plt.title("local energy")
				#plt.subplot2grid((1,2),(0,1))
				#plt.plot(X1.detach().numpy(),((Psi.detach().numpy()/max(np.abs(Psi.detach().numpy())))**2),ls='',marker='.')
				#plt.title("wavefunction")
				#plt.show()
				
			opt.zero_grad()  #delete remaining gradients
			J.backward()     #perform backward pass
			opt.step()       #update network parameters


			print("Progress {:2.0%}".format(step /steps), end="\r")  #some prints to monitor the training
		
		
		#compute energy of wavefunction after each epoch
		for n_samples in [10**1,10**1,10**1,10**1,10**1]:
			#n_samples = 10**6  #number of steps for monte carlo integration
			ts = time.time()
			samples=metropolis(lambda x :net(x)**2,(-6*np.array([1,1,1]),6*np.array([1,1,1])),np.array([0,0,0]),2,n_samples,presteps=500) #obtain samples
	
			#calculate energy
	
			X1 = samples[:,0]  
			X2 = samples[:,1]
			X3 = samples[:,2]
			X1.requires_grad=True
			X2.requires_grad=True
			X3.requires_grad=True
			X=torch.cat([X1,X2,X3], dim=0).reshape(3,n_samples).transpose(0,1)
	
			Psi=net(X).flatten() 
	
			dx =torch.autograd.grad(Psi,X1,create_graph=True,retain_graph=True,grad_outputs=torch.ones(n_samples))  
			ddx=torch.autograd.grad(dx[0].flatten(),X1,retain_graph=True,grad_outputs=torch.ones(n_samples))[0]
			dy =torch.autograd.grad(Psi,X2,create_graph=True,retain_graph=True,grad_outputs=torch.ones(n_samples))
			ddy=torch.autograd.grad(dy[0].flatten(),X2,retain_graph=True,grad_outputs=torch.ones(n_samples))[0]
			dz =torch.autograd.grad(Psi,X3,create_graph=True,retain_graph=True,grad_outputs=torch.ones(n_samples))
			ddz=torch.autograd.grad(dz[0].flatten(),X3,retain_graph=True,grad_outputs=torch.ones(n_samples))[0]
	
			lap_X = (ddx+ddy+ddz).flatten()
		

			r1    = torch.norm(X-R1,dim=1)  
			r2    = torch.norm(X-R2,dim=1)
			V     = -1/r1 - 1/r2 + 1/R
		
			E_loc_lap = -0.5*lap_X/Psi
			E    = (torch.mean(E_loc_lap*(E_loc_lap > 0).type(torch.FloatTensor) + V).item()*27.211386) # energy is given by mean of local energy over sampled batch from psi**2
			print('#samples = '+str(n_samples)+'    energyexpextation = '+str(E)+'    time = '+str(np.round((time.time()-ts),2)))

			plt.subplot2grid((1,2),(0,0))
			plt.plot(X1.detach().numpy(),(V).detach().numpy(),ls='',marker='.',label='Potential',color='y')
			plt.plot(X1.detach().numpy(),(-lap_X/Psi).detach().numpy(),ls='',marker='.',label='Laplacian/Psi',color='r')
			plt.plot(X1.detach().numpy(),(-lap_X/Psi + V).detach().numpy(),ls='',marker='.',label='Local energy (sum)',color='g')
			plt.legend(loc='lower center')


		print('___________________________________________')  #some prints to monitor the training
		print('It took', time.time()-start, 'seconds.')
		print('Energy = '+str(E))
		print('\n')


		if E < E_min:         #keep smallest energy and respective wavefunction
			E_min = E
			Psi_min = copy.deepcopy(net)

			
		#stuff for plotting the wavefunction along the x axis
		X_plot = torch.from_numpy(np.swapaxes(np.array([np.linspace(-6,6,100),np.zeros(100),np.zeros(100)]).reshape(3,100),0,1)).type(torch.FloatTensor)
		Psi_plot = net(X_plot).detach().numpy()
		plt.subplot2grid((1,2),(0,1))
		if epoch<(epochs-1) :
			plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),ls=':',color=cmap(epoch/epochs),linewidth =2)
			#plt.plot(X_plot[:,0].numpy(),(Psi_plot2/max(np.abs(Psi_plot2)))**2,label=str(np.round(E,2)),ls=':',color=cmap(epoch/epochs),linewidth =2)

		else:
			plt.plot(X_plot[:,0].numpy(),(Psi_plot/max(np.abs(Psi_plot)))**2,label=str(np.round(E,2)),color='k',linewidth =3)
		
		plt.legend()	
		plt.savefig(datetime.datetime.now().strftime("%B%d%Y%I%M%p")+".png")
	return (Psi_min,E_min)


for i in range(5):
	psi,e = fit(batch_size=200,steps=50000,epochs=5,losses=["energy"],R1=1,R2=-1)









