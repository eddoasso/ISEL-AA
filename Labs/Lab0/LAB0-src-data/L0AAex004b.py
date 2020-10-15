# -*- coding: latin-1 -*-
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
#visualizar  curva dada por: t=[-5pi,5pi]
# x=sin(t/2)*sin(t**2/10)  y=cos(t)  z=(t/20)**2
t=np.linspace(-np.pi,np.pi,1000)*5
x=np.sin(t/2)*np.sin(t**2/10)
y=np.cos(t);z=(t/20)**2
f1=plt.figure() #criar figura
ax=f1.add_subplot(111,projection='3d') #3D
ax.plot(x,y,z,'.-b') 
ax.elev=40   #posição da câmera:elevação
ax.azim=-92 #posição da câmera:azimute (em graus)
ax.set_xlabel('$x$',fontsize=16)
ax.set_ylabel('$y$',fontsize=16)
ax.set_zlabel('$z$',fontsize=16)
plt.savefig('../figs/L0AAex004b.png') #guardar em ficheiro ".png"
plt.show()
