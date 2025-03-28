"""
Author: Akankshya Mohanty
Mentor & Reviewer: Rajani Vanarse
#*******************************************************************
#Copyright (C) 2023 Adino Labs
#*******************************************************************
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm

#Subsetting the dataset
#Index 11856 marks the end of year 2013
df = pd.read_csv('Time_series_train.csv')
train=df[:] 

df2 = pd.read_csv('Time_Series_test.csv')
test=df2[:]

#Aggregating the dataset at daily level
#df.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 
#df.index = df.Timestamp 
#df = df.resample('D').mean()
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train.Timestamp 
train = train.resample('D').mean() 
test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
test.index = test.Timestamp 
test = test.resample('D').mean()

#Plotting data
#train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
#test.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
#plt.show()


#HOLT_WINTER
#y_hat_avg = test.copy()
#fit1 = ExponentialSmoothing(np.asarray(train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
#y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
#plt.figure(figsize=(16,8))
#plt.plot( train['Count'], label='Train')
#plt.plot(test['Count'], label='Test')
#plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
#plt.legend(loc='best')
#plt.show()


#rms = sqrt(mean_squared_error(test.Count, y_hat_avg.Holt_Winter))
#print(rms)



#ARIMA
y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
y_hat_avg['SARIMA'] = fit1.forecast(len(test))#fit1.predict(start="2014-01-01", end="2014-25-09", dynamic=True)
plt.figure(figsize=(16,8))
plt.plot( train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Count, y_hat_avg.SARIMA))
print(rms)
