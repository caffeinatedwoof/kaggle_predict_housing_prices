#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:37:20 2022

@author: admin
"""


# Compare Algorithms
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import os
#from xgboost import XGBRegressor
import preprocessing

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# load dataset
X_train, X_test, y_train, y_test = preprocessing.preprocess()
# prepare configuration for cross validation test harness
seed = 1
# prepare models
models = []
models.append(('LR', LinearRegression()))
models.append(('LS', Lasso()))
models.append(('RR', Ridge()))
#models.append(('XGB', XGBRegressor()))
# evaluate each model in turn
results = []
names = []
scoring = 'neg_mean_absolute_error'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, 
                                              cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
