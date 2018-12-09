from Combined import *

E_var1=[]
E_var2=[]
E_var3=[]

E_en1=[]
E_en2=[]
E_en3=[]

E_varen=[]

dist = [0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]

for R in dist:
    E_en1.append(fit(batch_size=2000,steps=5000,epochs=10,losses=["energy"],R1=R,R2=-R)[1])
    E_en2.append(fit(batch_size=500,steps=10000,epochs=10,losses=["energy"],R1=R,R2=-R)[1])
    E_en3.append(fit(batch_size=100,steps=20000,epochs=10,losses=["energy"],R1=R,R2=-R)[1])
    #E_en.append(fit(steps=2000,epochs=20,losses=["energy"],R1=R,R2=-R)[1])

plt.figure()
plt.plot(dist,E_var1,label="bs2000")
plt.plot(dist,E_var2,label="bs500")
plt.plot(dist,E_var3,label="bs100")
#plt.plot(dist,E_en)
plt.savefig("testlosses0912en.png")
E_var=np.mean(np.array([E_var1,E_var2,E_var3]),axis=0)
plt.plot(dist,E_var)
plt.savefig("testlossesmean0912en.png")

#plt.show()
