# -*- coding: latin-1 -*-
# ler MNIST em formato mat e pickle it
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

plt.close('all')

pName='MNISTsmall.p'
D=pickle.load(open(pName,'rb'))

X=D['X'].astype('float')
f1=D['foldTrain']
X=X[:,f1]

#tirar média
mx=np.mean(X,1)
Xn=X-mx[:,np.newaxis]
#Matriz de covariância
Cx=np.cov(Xn)
(v,W)=la.eig(Cx)
v=v.real
#ordenar v
idx=np.argsort(-v)
v=v[idx]
#ordernar W
W=W[:,idx]
#tirar valores pequenos
eps=np.finfo(float).eps #precisão da máquina
limiar=eps*v.max()
W=W[:,v>=limiar].real

print('%d componentes removidas'%(784-W.shape[1]))

#usar só as 3 primeiras componentes
#transformar: tb se podia ter feito Y=np.dot(W[:,0:3].T,Xn)
Y=np.dot(W.T,Xn)
#ver o variânica das comp. transformadas (Ys)
yVar=np.var(Y,1) #tem que ter +- mesmos valores que v (valores prórpios)
plt.figure(figsize=(7,5)) #criar figura
#usar logaritmo para ver valores altos e baixos no mesmo plot
plt.plot(np.log(yVar+eps)) 
plt.xlim(0,W.shape[1])
plt.grid()
plt.ylabel(r'$\log(\sigma_i^2+\epsilon)$',fontsize=16)
#normalizar Y de modo às variancias serem =1
Yn=np.dot(np.diag(np.std(Y,1)**-1),Y)
#construir indice de classes
classIdx=np.ones((10000))
for idx in np.arange(0,10):
    classIdx[idx*1000:(idx+1)*1000]=idx
    
f1=plt.figure(figsize=(7,5)) #criar figura
ax=f1.add_subplot(111,projection='3d') #3D
#5 primeiros dígitos
for i in np.arange(0,5):
    ax.plot(Y[0,classIdx==i],Y[1,classIdx==i],Y[2,classIdx==i],'.')

f1=plt.figure(figsize=(7,5)) #criar figura
ax=f1.add_subplot(111,projection='3d') #3D
#5 primeiros dígitos (dados normalizados)
for i in np.arange(0,5):
    ax.plot(Yn[0,classIdx==i],Yn[1,classIdx==i],Yn[2,classIdx==i],'.')

plt.show()
