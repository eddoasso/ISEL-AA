# -*- coding: latin-1 -*-
# ler MNIST e mostrar matrizes de covariância
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
    
def cmapGenerate(minV,maxV,N):
    #fazer cmap: minV<0, maxV>0
    #variação entre -V e +V com V=max(abs(minV),maxV)
    if np.abs(minV)>maxV:
        tstart=-1
        tend=maxV/np.abs(minV)
    else:
        tstart=minV/maxV
        tend=1
     
    #valor "0" corresponde a branco
    #-V azul, +V vermelho
    #criar cmap
    t=np.linspace(tstart,tend,N)
    sTmp=(1.0+np.cos(np.pi*t))/2.
    #Matriz com colormap
    CM=np.vstack((sTmp,sTmp,sTmp))
    #indice de t==0
    id0=sum(t<0)
    #1ª linha R
    t=np.linspace(0,tend,N-id0)
    sTmp=(3.0+np.cos(np.pi*t))/4.    
    CM[0,id0:]=sTmp
    #3ª linha B
    t=np.linspace(tstart,0,id0)
    sTmp=(3.0+np.cos(np.pi*t))/4.    
    CM[2,0:id0]=sTmp
    
    cmap = mpl.colors.ListedColormap(CM.T)
    return cmap



#################MAIN#####################################
plt.close('all')
pName='MNISTsmall.p'
D=pickle.load(open(pName,'rb'))
X=D['X']*1.
y=D['trueClass']
f1=D['foldTrain']
f2=D['foldTest']
X1=X[:,f1]
y1=y[f1]
X2=X[:,f2]
y2=y[f2]

for i  in range(10):
    dados=X1[:,y1==i]
    C=np.cov(dados)
    cmap=cmapGenerate(C.min(),C.max(),200)
    plt.figure(figsize=(5,5))   
    plt.imshow(C,cmap=cmap)
    plt.title('%d'%i)
    plt.box('on')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(shrink=.75)
plt.show()
