from Combined import *

E_var=[]
E_en=[]
E_varen=[]

dist = [0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]

for R in dist:
    E_var.append(fit(steps=2000,epochs=20,losses=["variance"],R1=R,R2=-R)[1])
    E_en.append(fit(steps=2000,epochs=20,losses=["energy"],R1=R,R2=-R)[1])
    
plt.figure()
plt.plot(dist,E_var)
plt.plot(dist,E_en)
plt.savefig("testlosses.png")
#plt.show()
