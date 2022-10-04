#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 15:10:30 2022

@author: admin
"""

#Import important packages
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pylab as py
from sklearn.linear_model import LinearRegression
import joblib
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../datasets', 'preprocessing'))
import preprocessing
#load data
X_train, X_test, y_train, y_test = preprocessing.preprocess()

#train our model
LR = LinearRegression()
LR.fit(X_train, y_train)

#test our model
result = LR.score(X_test, y_test)
print("RMSE is. {:.1f}".format(result))

#save our model in the model directory
model_name = "Linear Regression"
joblib.dump(LR, '../models/{}.pkl'.format(model_name))

#check residual plot
predictions = LR.predict(X_test)
residuals = (y_test - predictions)
sm.qqplot(residuals, line='45')
py.show()

#check homoscedasticity
plt.scatter(predictions, residuals)
plt.show()