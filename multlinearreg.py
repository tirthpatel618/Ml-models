import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# Download the data
df = pd.read_csv('/Users/tirthpatel/Desktop/Code/ML-models/Regression/FuelConsumption.csv')
cdf = df[['Engine size (L)','Cylinders','Combined (L/100 km)','CO2 emissions (g/km)']]

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['Engine size (L)','Cylinders','Combined (L/100 km)']])
y = np.asanyarray(train[['CO2 emissions (g/km)']])
regr.fit(x, y)
print('Coefficients: ', regr.coef_)

y_hat = regr.predict(test[['Engine size (L)','Cylinders','Combined (L/100 km)']])
x = np.asanyarray(test[['Engine size (L)','Cylinders','Combined (L/100 km)']])
y = np.asanyarray(test[['CO2 emissions (g/km)']])
print("Mean squared error: %.2f" % np.mean((y_hat - y) ** 2))

print("Variance score: %.2f" % regr.score(x, y))
