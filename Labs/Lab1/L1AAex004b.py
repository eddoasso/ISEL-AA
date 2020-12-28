# -*- coding: latin-1 -*-
import numpy as np
from matplotlib import pyplot as plt
#cov + média
plt.close('all')
A1=np.array([[1./3/np.sqrt(2.),1./np.sqrt(2.)],\
[-1./np.sqrt(2.)/3,1./np.sqrt(2.)]])
A2=np.array([[1./3/np.sqrt(2.),-1./np.sqrt(2.)],\
[1./np.sqrt(2.)/3,1./np.sqrt(2.)]])
b1=np.array([3,2]);b2=np.array([-3,2])
#criar 1000 pontos
N=1000;np.random.seed(0);X1=np.random.randn(2,N)
X2=(np.random.rand(2,N)-0.5)*np.sqrt(12.)#tirar média e por var=1
Y1=(np.dot(A1,X1).T+b1).T;Y2=(np.dot(A2,X2).T+b2).T
plt.figure(figsize=(7,4)) #criar figura
plt.plot(Y1[0,:],Y1[1,:],'.b',Y2[0,:],Y2[1,:],'.g',markersize=4)
plt.axis('scaled');plt.grid()
plt.xticks(np.arange(-6,6,1));plt.yticks(np.arange(-1,5,1))
plt.xlabel('$x_1$',fontsize=16);plt.ylabel('$x_2$',fontsize=16)
#plt.savefig('../figs/L1AAex004b.png',\
#bbox_inches='tight',transparent=True) #guardar em ficheiro ".png"
plt.show()
