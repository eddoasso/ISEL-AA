# -*- coding: latin-1 -*-
# geração de variáveis aleatórias 1D
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
x=rd.wald(1,3,int(1e5)) #1e5 pts aleatórios (mu=1,lambda=3)
#fazer histograma + plot da distribuição
hx,b=np.histogram(x,np.linspace(0,5,201),density=True)
#b->array de 201 entradas (fronteiras de quantificação)
#obter valores de quantificação (valor a meio dos intervalos)
b=(b[:-1]+b[1:])/2.0
t=np.linspace(0+1e-6,5,1000)
fx=np.sqrt(3/(2*np.pi*t**3))*np.exp(-3*(t-1)**2/(2*t))
plt.figure(figsize=(8,4))
plt.bar(b,hx,width=0.025,color=[0.9,.9,.9])
plt.plot(t,fx,'r',linewidth=2)
plt.axis([0,5,0,1.1])
plt.show()
#plt.savefig('../figs/L1AAex001.png',\
#bbox_inches='tight',transparent=True) 
