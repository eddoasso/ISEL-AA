import numpy as np
a=(np.arange(-6,6,2,float)+1).reshape((2,3))
print('a=',a ,'\nsoma=' , a.sum(),'\nproduto=',a.prod(),\
        '\nmedia=',a.mean(),'\nDesvio Padrao=',a.std())
#ou em alternativa
print('a=',a ,'\nsoma=' , np.sum(a),'\nproduto=',np.prod(a),\
        '\nmedia=',np.mean(a),'\nDesvio Padrao=',np.std(a))
b=np.array([[1,2,3],[4,3,2]],dtype=float)
print('a*b=',a*b,'\na**b=',a**b)
c=b[:,0]
print('\nc=',c)
#print('a+c',a+c ,'a*c',a*c)
#ERRO: a.shape=(2,3) e c.shape=(2,)
print('a+c=',a+c[:,np.newaxis],'\n a*c=',a*c[:,np.newaxis])
#funciona c[:,np.newaxis].shape=(2,1)
#c -> adicionada a cada coluna de a
