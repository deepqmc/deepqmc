from Combined import *


for i in range(10):
    fit(steps=1000,epochs=32,losses=["variance"])
    fit(steps=1000,epochs=32,losses=["energy"])
    fit(steps=1000,epochs=32,losses=["variance","energy"])
    fit(steps=1000,epochs=32)
