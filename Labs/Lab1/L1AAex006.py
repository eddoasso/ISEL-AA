# -*- coding: latin-1 -*-
import numpy as np
from matplotlib import pyplot as plt
import pickle
Parm=pickle.load(open('L1AAestrela.p','rb'))#importar pontos
pts0=Parm[0];pts1=Parm[1];pts2=Parm[2];pts3=Parm[3]
N=40 #nº de transformações
ang=np.arange(-np.pi,2.*np.pi,3.*np.pi/N)
transl=np.zeros((2,N))
transl[0,:]=np.arange(-1,1,2./N);transl[1,:]=transl[0,:]**2-1
scl=(1.+np.cos(transl[0,:]*np.pi/2+np.pi/2))-.8#escalamento
plt.figure(figsize=(7,6.5))    #criar figura
idx=0;media=np.zeros((2,1))    #vector temporario
for a in ang:
    T=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    media[0,0]=transl[0,idx];media[1,0]=transl[1,idx]
    s=scl[idx]
    idx=idx+1
    x0=np.dot(T,s*pts0)+media;x1=np.dot(T,s*pts1)+media
    x2=np.dot(T,s*pts2)+media;x3=np.dot(T,s*pts3)+media
    plt.plot(x0[0,:],x0[1,:],'-k',alpha=.1)
    plt.fill(x1[0,:],x1[1,:],'b',x2[0,:],x2[1,:],'g',\
    x3[0,:],x3[1,:],'r',alpha=.1)
plt.plot(x0[0,:],x0[1,:],'-k',linewidth=2)
plt.plot(x1[0,:],x1[1,:],'b',x2[0,:],x2[1,:],'g',x3[0,:],x3[1,:],'r')
plt.fill(x1[0,:],x1[1,:],'b',x2[0,:],x2[1,:],'g',x3[0,:],x3[1,:],'r')
plt.axis('scaled'),plt.grid(True),
plt.axis(np.array([-1.,1.,-1.,1.])*2)
plt.xlabel('$x_1$',fontsize=12),plt.xticks(np.arange(-2,2.1))
plt.ylabel('$x_2$',fontsize=12),plt.yticks(np.arange(-2,2.1))
#plt.savefig('../figs/L1AAex006.png',bbox_inches='tight',transparent=True) 
plt.show()
