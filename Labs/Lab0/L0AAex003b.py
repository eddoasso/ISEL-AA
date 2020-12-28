# -*- coding: latin-1 -*-
import numpy as np
from matplotlib import pyplot as plt
#gráfico de Gaussiana de média 0 (para dif.  variancias)
x=np.linspace(-5,5,1000) #1000pts equi-espaçados entre [-5,5]
sigma2=np.array([1./4,1./2,1.,2.,]) #valores da variancia
plt.figure()
for s in sigma2:
    fx=1/np.sqrt(2*np.pi*s)*np.exp(-1./(2*s)*x**2)
    plt.plot(x,fx)
    strTmp=r'$\sigma^2=$%0.2f'%s
    plt.text(0,fx[500],strTmp,fontsize=16)
plt.axis([-5,5,0,1])
plt.xticks(np.arange(-5,6))    #intervalos no x =  1
plt.yticks(np.arange(0,1,.1))  #intervalos no y =  0.1 
plt.grid()
plt.savefig('../figs/L0AAex003b.png') #guardar em ficheiro ".png"
plt.show()
