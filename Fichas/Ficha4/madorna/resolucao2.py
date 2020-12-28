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
from sklearn.metrics import confusion_matrix


dictionary = pickle.load(open("A45102_Q002_data.p","rb"))


x = dictionary['X']
y = dictionary['y']

x1 = -np.ones((2,2490))

wx = [0.0,0.98,0.19]

for i in range(2):
	for j in range(2490):
		x1[i,j] = wx[0]+wx[1]*x[0,j]+wx[2]*x[1,j]




Y1 = -np.ones((2,2490))

for i in range(2):
	Y1[i,y==i] = 1

#estimação a partir de um polinomio
values = np.vstack((np.zeros(x1.shape[1]),x1))
Rx = np.dot(values,values.T)
rxy = np.dot(values,Y1.T)
w = np.dot(la.pinv(Rx),rxy)


Y1e = np.dot(w.T,values)#Y1 estimado
y1e = np.argmax(Y1e,axis=0)

conf_matrix = confusion_matrix(y,y1e)
print("2-a:",conf_matrix)
print()


##### B ########


linReg = LinearRegression().fit(x.T,y)

y1e = linReg.predict(x.T)

mse2 = mean_squared_error(y, y1e)
print("2-b-i:",mse2)


w = linReg.coef_


x1 = -np.ones((2,2490))

wx = [0.0,w[0],w[1]]

for i in range(2):
	for j in range(2490):
		x1[i,j] = wx[0]+wx[1]*x[0,j]+wx[2]*x[1,j]




Y1 = -np.ones((2,2490))

for i in range(2):
	Y1[i,y==i] = 1


values = np.vstack((np.zeros(x1.shape[1]),x1))


Rx = np.dot(values,values.T)
rxy = np.dot(values,Y1.T)
wHand = np.dot(la.pinv(Rx),rxy)


Y1e = np.dot(wHand.T,values)#Y1 estimado
y1e = np.argmax(Y1e,axis=0)

conf_matrix = confusion_matrix(y,y1e)
print("2-b-ii:",conf_matrix)
print()


###### C #########


x1 = -np.ones((2,2490))

wx = [0.0,0.49,-0.87]

for i in range(2):
	for j in range(2490):
		x1[i,j] = wx[0]+wx[1]*x[0,j]+wx[2]*x[1,j]




Y1 = -np.ones((2,2490))

for i in range(2):
	Y1[i,y==i] = 1

#estimação a partir de um polinomio
values = np.vstack((np.zeros(x1.shape[1]),x1))
Rx = np.dot(values,values.T)
rxy = np.dot(values,Y1.T)
w = np.dot(la.pinv(Rx),rxy)


Y1e = np.dot(w.T,values)#Y1 estimado
y1e = np.argmax(Y1e,axis=0)

conf_matrix = confusion_matrix(y,y1e)
print("2-c-i:",conf_matrix)


