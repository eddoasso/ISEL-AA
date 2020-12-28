import sklearn.datasets as DT
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge,Lasso

D = DT.load_diabetes()

X=D.data
y=D.target

xtreino = X[:305][:]
xTeste = X[305:442][:]
yTreino = y[:305]
yTeste= y[305:442]

#EXERCICIO A
#alinea a TRUE
poly=PolynomialFeatures(degree=2,include_bias=False).fit(xtreino)
X1a=poly.transform(xtreino)
X2a=poly.transform(xTeste)

rl=LinearRegression().fit(X1a,yTreino)

print('a i',rl.score(X2a,yTeste))
#alinea b FALSE

y2e=rl.predict(X2a)
print("a ii ",mean_squared_error(yTeste,y2e))


#EXERCICIO B
#ALINEA A

w=rl.coef_ #qnd se usa ridge os coefecientes nunca chegam a 0
plt.plot(w)
plt.show()
print(rl.intercept_)
print("numeroCoefecientes",len(w))

#alinea B



#EXERCICIO B

poly=PolynomialFeatures(degree=4,include_bias=False).fit(xtreino)
X1a=poly.transform(xtreino)
X2a=poly.transform(xTeste)

rl=LinearRegression().fit(X1a,yTreino)
w=rl.coef_ #qnd se usa ridge os coefecientes nunca chegam a 0
plt.plot(w)
plt.show()
print(rl.intercept_) 
print("numeroCoefecientes",len(w)) #1000 





#ALINEA C   TRUE

poly=PolynomialFeatures(degree=3,include_bias=False).fit(xtreino)
X1a=poly.transform(xtreino)
X2a=poly.transform(xTeste)

rl=Lasso(alpha=0.01,random_state=42).fit(X1a,yTreino) # da me valores dos coefecientes igual a 0
print('R2 (treino:',rl.score(X1a,yTreino))
print('R2 (teste) :',rl.score(X2a,yTeste))

y1e=rl.predict(X1a)
print("MEAN SQUARED ERROR",mean_squared_error(yTreino,y1e))



w=rl.coef_ #exclui o zero
print("c ii ", np.sum(np.abs(w)!= 0))


