# -*- coding: latin-1 -*-
# ler MNIST em formato mat e pickle it
# LDA: Separar "0"s ,"1"s..., "4"s
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

plt.close('all')

pName='MNISTsmall.p'
D=pickle.load(open(pName,'rb'))
X=D['X']*1.
y=D['trueClass']
f1=D['foldTrain']
X=X[:,f1]
y=y[f1]
X=np.hstack((X[:,y==0],X[:,y==1],X[:,y==2],X[:,y==3],X[:,y==4]))

preProcFlag=True#False #
###############################################################
# Pré-Processar dados com PCA antes de fazer FISHER
# Necessário para evitar problemas numéricos:
# Caso não se faça o pré-processamento a função la.inv() DÁ ERROS!
# Ao usar la.pinv (pseudo-inverso) já funciona
###############################################################
if preProcFlag:
    #tirar média
    mx=np.mean(X,1)
    Xn=X-mx[:,np.newaxis]
    #Matriz de covariância
    Cx=np.cov(Xn)
    (v,V)=la.eig(Cx)
    v=v.real
    #ordenar v
    idx=np.argsort(-v)
    v=v[idx]
    #ordernar W
    V=V[:,idx]
    #tirar valores pequenos
    eps=np.finfo(float).eps #precisão da máquina
    #limiar=eps*v.max()
    limiar=1.e-10
    V=V[:,v>=limiar].real
    
    print('%d componentes removidas'%(784-V.shape[1]))
    
    #transformar: 
    Y=np.dot(V.T,Xn)
else:
    #Não há pré-processamento
    Y=X

#FISHER Calcular médias e covariâncias por classe
mtot=np.mean(Y,1) #media total
mtot=mtot[:,np.newaxis] #para mtot (dx1)
m0=np.mean(Y[:,0:1000],1)
m0=m0[:,np.newaxis] #para m0 (dx1)
m1=np.mean(Y[:,1000:2000],1)
m1=m1[:,np.newaxis]
m2=np.mean(Y[:,2000:3000],1)
m2=m2[:,np.newaxis]
m3=np.mean(Y[:,3000:4000],1)
m3=m3[:,np.newaxis]
m4=np.mean(Y[:,4000:5000],1)
m4=m4[:,np.newaxis]
C0=np.cov(Y[:,0:1000])
C1=np.cov(Y[:,1000:2000])
C2=np.cov(Y[:,2000:3000])
C3=np.cov(Y[:,3000:4000])
C4=np.cov(Y[:,4000:5000])

Smu=np.dot((m0-mtot),(m0-mtot).T)+np.dot((m1-mtot),(m1-mtot).T)+\
np.dot((m2-mtot),(m2-mtot).T)+np.dot((m3-mtot),(m3-mtot).T)\
+np.dot((m4-mtot),(m4-mtot).T)
Cs=C0+C1+C2+C3+C4
#decompor em valores e vectores prórpios
#matriz total
M=np.dot(la.pinv(Cs),Smu)
#ordenar e escolher os 4 (nº classes -1) vectores próprios
#associados aos maiores valores  próprios
(v2,V2)=la.eig(M)
v2=v2.real
idx2=np.argsort(-v2)
#matrix de transformação
W=V2[:,idx2[0:4]].real

#z dados 4d projectados da direcção de Fisher
Z=np.dot(W.T,Y)

#visualizar os Z (dados 4D)
#construir indice de classes
classIdx=np.ones((5000))
for idx in np.arange(0,5):
    classIdx[idx*1000:(idx+1)*1000]=idx
    
f1=plt.figure(figsize=(7,5)) #criar figura
ax=f1.add_subplot(111,projection='3d') #3D
#5 primeiros dígitos
for i in np.arange(0,5):
    ax.plot(Z[0,classIdx==i],Z[1,classIdx==i],Z[2,classIdx==i],'.',ms=2)
ax.azim=-60
ax.elev=20
ax.set_xlabel(r'$x_1$',fontsize=16)
ax.set_ylabel(r'$x_2$',fontsize=16)
ax.set_zlabel(r'$x_3$',fontsize=16)

f1=plt.figure(figsize=(7,5)) #criar figura
ax=f1.add_subplot(111,projection='3d') #3D
#5 primeiros dígitos
for i in np.arange(0,5):
    ax.plot(Z[0,classIdx==i],Z[1,classIdx==i],Z[3,classIdx==i],'.',ms=2)
ax.azim=-60
ax.elev=20
ax.set_xlabel(r'$x_1$',fontsize=16)
ax.set_ylabel(r'$x_2$',fontsize=16)
ax.set_zlabel(r'$x_4$',fontsize=16)

plt.show()
