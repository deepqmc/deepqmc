from ELU import *

E_var1=[]
E_var2=[]
E_var3=[]

E_en1=[]
E_en2=[]
E_en3=[]

E_varen=[]

#dist = [0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
dist = [1.0]

for R in dist:
    E_en1.append(fit(batch_size=2000,steps=5000,epochs=10,losses=["energy","symmetry","energy"],R1=R,R2=-R)[1])
    #E_en.append(fit(steps=2000,epochs=20,losses=["energy"],R1=R,R2=-R)[1])

#plt.figure()
#plt.plot(dist,E_en1)
#plt.savefig("testlosses0912en.png")

plt.show()
