import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# Download the data
df = pd.read_csv('/Users/tirthpatel/Desktop/Code/ML-models/Regression/FuelConsumption.csv')

cdf = df[['Engine size (L)','Cylinders','Combined (L/100 km)','CO2 emissions (g/km)']]
print(cdf.head(9))  

# Plotting the data
viz = cdf[['Engine size (L)','Cylinders','Combined (L/100 km)','CO2 emissions (g/km)']]
viz.hist()
plt.show()

plt.scatter(cdf.Cylinders, cdf['CO2 emissions (g/km)'],  color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
plt.scatter(train['Engine size (L)'], train['CO2 emissions (g/km)'],  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Engine size (L)']])
train_y = np.asanyarray(train[['CO2 emissions (g/km)']])
regr.fit(train_x, train_y)

# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ',regr.intercept_)
plt.scatter(train['Engine size (L)'], train['CO2 emissions (g/km)'],  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Evaluation
test_x = np.asanyarray(test[['Engine size (L)']])
test_y = np.asanyarray(test[['CO2 emissions (g/km)']])
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_))


#fuel consumption metric
train_x = train[['Combined (L/100 km)']]
test_x = test[['Combined (L/100 km)']]
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
predictions = regr.predict(test_x)
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))
