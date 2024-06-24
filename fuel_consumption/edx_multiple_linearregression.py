import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt

#loading dataset
df=pd.read_csv('E:\\datascience\\ed_x\\ed_x_project_and_lab\\csv\\FuelConsumptionCo2.csv')
df.head()

#selecting features
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#ploting emission values wrt engine size
plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()

#train and test data
msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

#train data distribution
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()

#multiple regression model
from sklearn import linear_model
regr=linear_model.LinearRegression()
x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
print('Coefficients:',regr.coef_)
#Given that it is a multiple linear regression model with 3 parameters and that the parameters are the intercept and coefficients of the hyperplane,
#sklearn can estimate them from our data. 
#Scikit-learn uses plain Ordinary Least Squares method to solve this problem.

#prediction
y_hat=regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y=np.asanyarray(test[['CO2EMISSIONS']])
print('Mean Squared Error (MSE) : %.2f'%np.mean((y_hat-y)**2))
# Explained variance score: 1 is perfect prediction
print('Varience score: %.2f'%regr.score(x,y))

#multiple LR with different attributes
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)
y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))