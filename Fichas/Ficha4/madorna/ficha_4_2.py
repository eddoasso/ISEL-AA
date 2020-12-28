import sklearn.datasets as DT
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge,Lasso
import pickle
from sklearn.metrics import mean_absolute_error

D=pickle.load(open('C:/Users/Asus/Desktop/ISEL/AA/AA/A45140_Ficha4_ficheiros/A45140_Q002_data.p','rb'))

print(D.keys())


X = D['x']
y = D['y']
folds = D['folds']

train_x = X[folds==0][:,np.newaxis]
test_x = X[folds==1][:,np.newaxis]

train_y = y[folds==0][:,np.newaxis]
test_y = y[folds==1][:,np.newaxis]

#PERGUNTA A # deu mal tudo XD

poly=PolynomialFeatures(degree=3,include_bias=False).fit(train_x)
X1a=poly.transform(train_x)
X2a=poly.transform(test_x)
rl=LinearRegression().fit(X1a,train_y)


print('a i',rl.score(X2a,test_y))




#PERGUNTA C

poly=PolynomialFeatures(degree=4,include_bias=False).fit(train_x)
X1a=poly.transform(train_x)
X2a=poly.transform(test_x)
rl=LinearRegression().fit(X1a,train_y)

y1e=rl.predict(X1a)
print("c i ", mean_squared_error(train_y,y1e))  #6,86



y2e=rl.predict(X2a)
print("c ii ", mean_absolute_error(test_y,y2e))  #6,86


#PERGUNTA B sobre aprendizagem XDDDDDDDDDDDDDDDDDDDDDDD

train_x1 = X[:130][:,np.newaxis]

train_y1 = y[:130]

train_x2 = X[130:][:,np.newaxis]

train_y2 = y[130:]


polyA = PolynomialFeatures(degree=4,include_bias=False).fit(train_x1)

poly_trainA = polyA.transform(train_x1)

poly_trainAx = polyA.transform(train_x2)

ya = LinearRegression().fit(poly_trainA,train_y1)

y1e = ya.predict(poly_trainAx)

mse = mean_squared_error(train_y2, y1e)

print(mse)



polyB = PolynomialFeatures(degree=4,include_bias=False).fit(test_x)

poly_trainB = polyB.transform(test_x)

poly_trainBx = polyB.transform(train_x)

yb = LinearRegression().fit(poly_trainB,test_y)

y2e = ya.predict(poly_trainBx)

mse2 = mean_squared_error(train_y, y2e)

print(mse2)

print("valor maximo:",X.max())
print(yb.score(poly_trainB,test_y))
print(yb.score(poly_trainBx,train_y))