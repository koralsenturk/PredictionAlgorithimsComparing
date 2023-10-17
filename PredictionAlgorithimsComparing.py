# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar_2odev.csv')

#ünvan verisinin sayısal karşılığı verildiği için label encoding yapılmayacak.
x = veriler.iloc[:,2:5]  #2:3 olarak yapılınca daha verimli
y = veriler.iloc[:,5:]
X = x.values
Y = y.values 


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)




import statsmodels.api as sm
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

# sadece 2.kolonun kullanılması fikri p value değerleri sonrası oluşmuştur, buna bağlı olarak kolon seçimleri yapılmıştır.


#polynomial regression


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)





#tahminler



model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())


#verilerin olceklenmesi

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)






from sklearn.svm import SVR

svr_reg = SVR(kernel ='rbf')
svr_reg.fit(x_olcekli, y_olcekli)




model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())




from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)






model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())





from sklearn.ensemble import RandomForestRegressor

r_rf = RandomForestRegressor(random_state=0, n_estimators =10)
r_rf.fit(X, Y.ravel())



model5 = sm.OLS(r_rf.predict(X),X)
print(model5.fit().summary())





from sklearn.metrics import r2_score
print('Random Forest r2 değeri')
print(r2_score(Y, r_rf.predict(X)))


print('Decision Tree r2 değeri')
print(r2_score(Y, r_dt.predict(X)))


print('Support vector regression r2 değeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


print('Polinomil regression r2 değeri')
print(r2_score(Y, lin_reg2.predict(x_poly)))


print('Linear regression r2 değeri')
print(r2_score(Y, lin_reg.predict(X)))



print(veriler.corr())