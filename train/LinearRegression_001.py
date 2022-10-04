#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 15:10:30 2022

@author: admin
"""

#Import important packages
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
import sys, os
import preprocessing
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../datasets', 'preprocessing'))

#Log Experimental Run

experimental_run = '001'

#load data
X_train, X_test, y_train, y_test = preprocessing.preprocess()

#train our model
LR = LinearRegression()
LR.fit(X_train, y_train)

#test our model
result = LR.score(X_test, y_test)
print("RMSE is. {:.1f}".format(result))

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
logging.basicConfig(filename='../train/LinearRegression001.log', encoding='utf-8',
                    level=logging.INFO)
logging.info("RMSE is. {:.3f}".format(result))