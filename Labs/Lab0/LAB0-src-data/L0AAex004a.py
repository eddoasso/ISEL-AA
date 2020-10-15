# -*- coding: latin-1 -*-
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
#visualizar função f(x,y)=sin(x+x*y)*sin(y+x*y)/(x+x*y)/x
#criar grelha com função meshgrid
X,Y=np.meshgrid(np.linspace(-5,5.,50),np.linspace(-3,3.,50))
#X e Y arrays bi-dimensionais de 50x50
Z=np.sin(X+X*Y)*np.sin(Y+X*Y)/(Y+X*Y)/X
f1=plt.figure() #criar figura
ax=f1.add_subplot(111,projection='3d') #3D
ax.plot_wireframe(X,Y,Z,alpha=.3)
ax.contour(X,Y,Z,25,offset=Z.min()) #25=nº contornos
ax.elev=20   #posição da câmera:elevação
ax.azim=-150 #posição da câmera:azimute (em graus)
ax.set_xlabel('$x$',fontsize=16)
ax.set_ylabel('$y$',fontsize=16)
ax.set_zlabel('$f(x,y)$',fontsize=16)
plt.savefig('../figs/L0AAex004a.png') #guardar em ficheiro ".png"
plt.show()
