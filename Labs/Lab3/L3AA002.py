# -*- coding: latin-1 -*-
# ler MNIST em formato mat e pickle it
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.close('all')

pName='MNISTsmall.p'
D=pickle.load(open(pName,'rb'))
X=D['X']*1.
y=D['trueClass']
f1=D['foldTrain']
X1=X[:,f1]
y1=y[f1]
#matriz com AND
Cand=np.ones((28,28)).astype('bool')
#matriz com OR
Cor=np.zeros((28,28)).astype('bool')
for i in range(10):
    dados=X1[:,y1==i]
    C=np.cov(dados)
    #matriz (boolean)
    I0=np.reshape(np.diag(C)!=0,(28,28))
    Cand=Cand&I0
    Cor=Cor|I0
    #print '(%d) %d\n'%(contador,np.sum(I0==False))
    plt.figure(figsize=(5,5))   
    plt.imshow(I0,cmap='gray',interpolation='none')
    plt.box('on')
    plt.title('%d'%i)
    plt.xticks(np.arange(0,28)-.5,'')
    plt.yticks(np.arange(0,28)-.5,'')
    plt.grid('on')

plt.figure(figsize=(5,5))   
plt.imshow(Cand,cmap='gray',interpolation='none')
plt.box('on')
plt.xticks(np.arange(0,28)-.5,'')
plt.yticks(np.arange(0,28)-.5,'')
plt.grid('on')
plt.title('AND')

plt.figure(figsize=(5,5))   
plt.imshow(Cor,cmap='gray',interpolation='none')
plt.title('OR')
plt.box('on')
plt.xticks(np.arange(0,28)-.5,'')
plt.yticks(np.arange(0,28)-.5,'')
plt.grid('on')
plt.show()
