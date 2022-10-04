#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 15:17:54 2022

@author: admin
"""

#Import important packages
import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.path.dirname(__file__), '../datasets', 'preprocessing'))
import preprocessing
import numpy as np 
import xgboost as xgb
import joblib
from sklearn.model_selection import RandomizedSearchCV, KFold


#load data
X_train, X_test, y_train, y_test = preprocessing.preprocess()

#train our model
#xgb = xgboost.XGBRegressor(n_estimators=1000, max_depth=2, eta=0.1, subsample=0.7, colsample_bytree=0.8)
#xgb.fit(X_train, y_train)

model = xgb.XGBRegressor()
# A parameter grid for XGB
params = {
    'learning_rate': [.01, .03, 0.05, .07],
    'min_child_weight': [4, 5],
    'gamma': [i / 10.0 for i in range(3, 6)],
    'subsample': [i / 10.0 for i in range(6, 11)],
    'colsample_bytree': [i / 10.0 for i in range(6, 11)],
    'max_depth': [2, 3, 4]
}

#test our model
#result = xgb.score(X_train, y_train)
#print("RMSE is. {:.1f}".format(result))

#save our model in the model directory
#model_name = "XGB Regressor"
#joblib.dump(xgb, '../models/{}.pkl'.format(model_name))

# Initialize XGB and RandomSearch
folds = 10
param_comb = 500

cv = KFold(n_splits=folds, shuffle=True, random_state = 1)

random_search = RandomizedSearchCV(model, param_distributions=params, 
                                   n_iter=param_comb, 
                                   scoring = 'neg_mean_squared_error', 
                                   n_jobs=-1, cv=cv, 
                                   random_state=1 )

# Here we go
#start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_train, y_train)
#timer(start_time) # timing ends here for "start_time" variable

#print('\n All results:')
#print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized root mean squared error for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(np.sqrt(random_search.best_score_))
print('\n Best hyperparameters:')
print(random_search.best_params_)


def build_XGB(
    _subsample=0.1, 
    _min_child_weight=4, 
    _max_depth=4, 
    _learning_rate=0.07, 
    _gamma=0.3, 
    _colsample_bytree=1.0):
    model = xgb.XGBRegressor(subsample=_subsample, min_child_weight=_min_child_weight, 
                         max_depth=_max_depth, learning_rate=_learning_rate, gamma=_gamma, 
                             colsample_bytree=_colsample_bytree)
    return model
# define the model
model = build_XGB()
# define the datasets to evaluate each iteration
evalset = [(X_train, y_train), (X_test,y_test)]
# fit the model
model.fit(X_train, y_train, eval_set=evalset,verbose=False)
# evaluate performance
yhat = model.predict(X_test)
#score = np.sqrt(mean_squared_error(yhat, y_test))
#print('RMSE: %.3f' % score)
# retrieve performance metrics
results = model.evals_result()
# plot learning curves
plt.plot(results['validation_0']['rmse'], label='train')
plt.plot(results['validation_1']['rmse'], label='cross-validation')
# show the legend
plt.legend()
# show the plot
plt.show()