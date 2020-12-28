# -*- coding: latin-1 -*-
import numpy as np
from matplotlib import pyplot as plt
#criar matrix A 2x2 e vector b 2x1
A=np.array([[1./3/np.sqrt(2.),1./np.sqrt(2.)],\
[-1./np.sqrt(2.)/3,1./np.sqrt(2.)]])
b=np.array([3,2])
#criar 1000 pontos
np.random.seed(0);X=np.random.randn(2,1000)
#para somar b, tem que se transpor x (e voltar a transpor)
Y=(np.dot(A,X).T+b).T
#ou em alternativa: Y=np.dot(A,X)+b[:,np.newaxis]
plt.figure(figsize=(7,5)) #criar figura
plt.plot(X[0,:],X[1,:],'.b',markersize=4)
plt.plot(Y[0,:],Y[1,:],'og',markersize=4)
plt.axis('equal');plt.axis([-5.,7,-4.,5.1])
plt.grid();plt.xticks(np.arange(-4,7))
plt.xlabel('$x_1$',fontsize=16)
plt.ylabel('$x_2$',fontsize=16)
#plt.savefig('../figs/L1AAex003.png',\
#box_inches='tight',transparent=True) #guardar em ficheiro ".png"
plt.show()
