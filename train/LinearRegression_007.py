#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 15:10:30 2022

@author: admin
"""

#Import important packages
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../datasets', 'preprocessing'))
import preprocessing
import logging

#Log Experimental Run

experimental_run = '007'

#path to the dataset
filename = "../datasets/train.csv"

#Read CSV File
train = pd.read_csv(filename, index_col='Id')

#load data
X_train, X_test, y_train, y_test = preprocessing.preprocess(train)

#train our model
LR = LinearRegression()
LR.fit(X_train, y_train)

#test our model
result = mean_squared_error(y_test, LR.predict(X_test), squared=False)
print("RMSE is. {:.3f}".format(result))

#save our model in the model directory
joblib.dump(LR, '../artifacts/train/LinearRegression{}.pkl'.format(experimental_run))

#check residual plot
predictions = LR.predict(X_test)
residuals = (y_test - predictions)
sm.qqplot(residuals, line='45')
plt.show()
plt.savefig('../artifacts/train/residualplotLR{}.png'.format(experimental_run))

#check homoscedasticity
plt.scatter(predictions, residuals)
plt.show()
plt.savefig('../artifacts/train/homoscedasticityLR{}.png'.format(experimental_run))

#log experimental results
logging.basicConfig(filename='../train/LinearRegression{}.log'.format(experimental_run), 
                    encoding='utf-8',
                    level=logging.INFO)
logging.info("RMSE is. {:.3f}".format(result))