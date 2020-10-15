# -*- coding: latin-1 -*-
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
#Gerar pontos 3D (três grupos com 5000 pts cada)
x1=np.random.randn(3,5000)*1/2.+np.array([[2],[0],[2]])
x2=np.random.rand(3,5000)*2+np.array([[0],[2],[2]])
A3=np.array([[1,.5,0],[.1,.0,.1],[.1,1,.1]])
x3=A3.dot(x1)-np.array([[-2],[1],[-2]])
f1=plt.figure() #criar figura
ax=f1.add_subplot(111,projection='3d') #3D
ax.plot(x1[0,:],x1[1,:],x1[2,:],'ob',markersize=3) 
ax.plot(x2[0,:],x2[1,:],x2[2,:],'sr',markersize=3) 
ax.plot(x3[0,:],x3[1,:],x3[2,:],'^g',markersize=3) 
ax.elev=20   #posição da câmera:elevação
ax.azim=-140 #posição da câmera:azimute (em graus)
ax.set_xlabel('$x$',fontsize=16)
ax.set_ylabel('$y$',fontsize=16)
ax.set_zlabel('$z$',fontsize=16)
ax.set_aspect('equal','box')  #todos eixo à mesma escala
#plt.savefig('../figs/L0AAex004c.png',\
#bbox_inches='tight',transparent=True) 
plt.show()
