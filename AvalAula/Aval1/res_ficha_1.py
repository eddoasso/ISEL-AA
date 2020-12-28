import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


#1
X=rd.rand(2,10000)*2-1

plt.plot(X[0,:],X[1,:],'.',ms=1)
plt.grid(True)
plt.axis('scaled')

#2

A=np.array([[1.5,0],[0,0.75]])
X2=np.dot(A,X)
print(X2.shape)
a=np.pi/6
B=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
X3= np.dot(B,X2)
plt.plot(X3[0,:],X3[1,:],'.',ms=1)
print(X3.shape)


#3

h0,b0 =np.histogram(X3[0,:],np.linspace(-2,2,500),density=True)
b0 =b0[:-1]+b0[1:]/2
h1,b1=np.histogram(X3[0,:],np.linspace(-1.5,1.5,500),density=True)
b1=b1[:-1]+b1[1:]/2
plt.bar(b0,h0)
    
#pedro is very sexy 
