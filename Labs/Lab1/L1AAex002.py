# -*- coding: latin-1 -*-
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
plt.close('all')
N=100000 #1e5 pts uniforme
x1=rd.rand(N)
x2=np.sum(rd.rand(2,N),axis=0) #soma de 2 v.a. uniformes
x3=np.sum(rd.rand(3,N),axis=0) #soma de 3 v.a. uniformes
x4=np.sum(rd.rand(10,N),axis=0) #soma de 10 v.a. uniformes
hx,b=np.histogram(x1,np.linspace(0,1,101),density=True)
b=(b[:-1]+b[1:])/2
plt.figure(figsize=(15,5))
plt.subplot(141);plt.axis([-0.1,1.1,0,1.1]);plt.title('N=1')
plt.bar(b[0:100],hx,width=0.025);plt.xticks([0,0.5,1])
hx,b=np.histogram(x2,np.linspace(0,2,101),density=True)
b=(b[:-1]+b[1:])/2
plt.subplot(142);plt.axis([-0.1,2.1,0,1.1]);plt.title('N=2')
plt.bar(b[0:100],hx,width=0.025);plt.xticks(np.arange(3))
hx,b=np.histogram(x3,np.linspace(0,3,101),density=True)
b=(b[:-1]+b[1:])/2
plt.subplot(143);plt.axis([-0.1,3.1,0,.8]);plt.title('N=3')
plt.bar(b[0:100],hx,width=0.05);plt.xticks(np.arange(4))
hx,b=np.histogram(x4,np.linspace(0,10,101),density=True)
b=(b[:-1]+b[1:])/2
plt.subplot(144);plt.axis([-0.1,10.1,0,.5]);plt.title('N=10')
plt.bar(b[0:100],hx,width=0.1);plt.xticks(np.arange(0,11,2))
#plt.savefig('../figs/L1AAex002.png',bbox_inches='tight',transparent=True) 
plt.show()
