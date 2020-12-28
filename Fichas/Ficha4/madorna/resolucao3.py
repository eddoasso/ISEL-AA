import numpy as np
from matplotlib import pyplot as plt
import pickle
import sklearn.datasets as ds
import sklearn.preprocessing as pp
import scipy as sc
from sklearn.decomposition import PCA
import scipy.linalg as la
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


dictionary = pickle.load(open("A45102_Q003_data.p","rb"))


X = dictionary['x']
y = dictionary['y']
folds = dictionary['folds']


train_x = X[folds==1][:,np.newaxis]

test_x = X[folds==0][:,np.newaxis]

train_y = y[folds==1]

test_y = y[folds==0]




poly = PolynomialFeatures(degree=3,include_bias=False).fit(train_x)

poly_train = poly.transform(train_x)
poly_test = poly.transform(test_x)

linReg = LinearRegression().fit(poly_train,train_y)

y1e = linReg.predict(poly_test)

mse = mean_squared_error(test_y, y1e)

print("3-a-i:",mse)

y2e = linReg.predict(poly_train)

msa = mean_absolute_error(train_y, y2e)

print("3-a-ii:",msa)





####### B #######

poly = PolynomialFeatures(degree=4,include_bias=False).fit(train_x)

poly_train = poly.transform(train_x)
poly_test = poly.transform(test_x)

linReg = LinearRegression().fit(poly_train,train_y)


w0 = linReg.intercept_

print("3-b-i:",w0)

y1e = linReg.predict(poly_test)

msa = mean_absolute_error(test_y, y1e)

print("3-b-ii:",msa)
print()


####### C #######


train_x1 = X[:53][:,np.newaxis]

train_y1 = y[:53]

train_x2 = X[53:][:,np.newaxis]

train_y2 = y[53:]


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


