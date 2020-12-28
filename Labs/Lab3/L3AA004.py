import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle
# Discriminantes de Fisher Separar "0"s e "1"s

plt.close('all')

pName='MNISTsmall.p'
D=pickle.load(open(pName,'rb'))
X=D['X']*1.
y=D['trueClass']
f1=D['foldTrain']
X=X[:,f1]
y=y[f1]
X=np.hstack((X[:,y==0],X[:,y==1]))

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
    
    #transformar: 
    Y=np.dot(W.T,Xn)
else:
    #Não há pré-processamento
    Y=X

#FISHER Calcular médias e covariâncias por classe
m0=np.mean(Y[:,0:1000],1)
m1=np.mean(Y[:,1000:],1)
C0=np.cov(Y[:,0:1000])
C1=np.cov(Y[:,1000:])

wFisher=np.dot(la.inv(C0+C1),m0-m1)
#z dados 1d projectados da direcção de Fisher
z=np.dot(wFisher.T,Y)

#calcular médias de z (projectar os m0 e m1)
mz0=np.dot(wFisher.T,m0)
mz1=np.dot(wFisher.T,m1)
#Limiar optimo (a meio de mz0 e mz1)
zLimiar=(mz0+mz1)/2.0
print('Limiar de Decisão: %.3f'%zLimiar)
(hz,bz)=np.histogram(z,bins=100,density=True)
#bz -> dimensões (101,)
#hz -> dimensões (100,)
# converter bz para (100,)
bnew=(bz[0:-1]+bz[1:])/2.0

#visualizar os z (dados 1D)
plt.figure(figsize=(7,3.5)) #criar figura
plt.plot(z,'.',color=[.1,.1,.5])
plt.plot([0,2000],[zLimiar,zLimiar],color=[.2,.5,.2],lw=2)
#há 2 erros -> ver quais
Erros0=z[0:1000]<zLimiar
Erros1=z[1000:]>zLimiar
print('Número de erros nos "0": %d'%np.sum(Erros0))
print('Número de erros nos "1": %d'%np.sum(Erros1))
#índice dos erros0
(idx0,)=np.nonzero(Erros0)
#índice dos erros1
(idx1,)=np.nonzero(Erros1)
idx1=idx1+1000 #idx1 começa a 1000
#fazer plot dos erros
plt.plot(idx0,z[idx0],'o',color=[.9,.1,.1],ms=9,alpha=.4)
plt.plot(idx1,z[idx1],'s',color=[.9,.1,.1],ms=9,alpha=.4)

plt.grid()
plt.axis([0,2000,-90,90])
plt.yticks(np.arange(-80,90,20))

#fazer histograma dos z
plt.figure(figsize=(7,3.5)) #criar figura
plt.bar(bnew,hz,width=1.25,color=[.7,.7,.7])
plt.plot([zLimiar,zLimiar],[0.,0.05],color=[.2,.5,.2],lw=3)
plt.grid()
plt.axis([-90,90,0,0.05])
plt.xticks(np.arange(-80,90,20))

#duas imagens com erros: indice 527 (0), indice 1127 (1)
DEr0=X[:,527]
DEr1=X[:,1127]
plt.figure(figsize=(5,5))
d0=np.reshape(DEr0,(28,28))
plt.imshow(255-d0,cmap='gray',interpolation='none')
plt.xticks([])
plt.yticks([])
plt.box('on')

plt.figure(figsize=(5,5))
d1=np.reshape(DEr1,(28,28))
plt.imshow(255-d1,cmap='gray',interpolation='none')
plt.xticks([])
plt.yticks([])
plt.box('on')

plt.show()
