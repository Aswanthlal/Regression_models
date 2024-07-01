import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#loading dataset
df=pd.read_csv('FuelConsumptionCo2.csv')
df.head()

#summerizing the data
df.describe()

#selecting features
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#ploting each features
viz=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()

#ploting eCH feature against emission to see their relationship 
plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('EMISSION')
plt.show()

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('Cylinders')
plt.ylabel('Emission')
plt.show()


#Creating and training a dataet
#80% of the entire dataset will be used for training and 20% for testing
msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

#Simple regression model

#train data distribution
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#modeling
from sklearn import linear_model
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print('Coefficients:',regr.coef_)
print('Intercept:',regr.intercept_)

#ploting outputs
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0],'-r')
plt.xlabel('ENGINESIZE')
plt.ylabel('EMISSION')
plt.show()

#evaluation
#compare the actual values and predicted values to calculate the accuracy of a regression model
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_))


#Predict data and find MAE value
train_x=train[['FUELCONSUMPTION_COMB']]
test_x=test[['FUELCONSUMPTION_COMB']]

regr=linear_model.LinearRegression()
regr.fit(train_x,train_y)
predictions=regr.predict(test_x)
print('Mean absolute error:%.2f'%np.mean(np.absolute(predictions-test_y)))
