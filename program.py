#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:17:19 2022

@author: admin
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys
import pickle
import os
import yaml

# folder to load config file
CONFIG_PATH = "../config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
        
        return config

# predict fitted values using trained models and print score
def run_model(X_test, y_test, model):
    predictions = model.fit(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print('RMSE is ' +str(rmse))
    
def main():
    config = load_config("my_config.yaml")

    # load data
    data = pd.read_csv(os.path.join(config["data_directory"], config["data_name"]),
                       index_col = config["index_col"])

    # Define X (independent variables) and y (target variable)
    X = np.array(data.drop(config["target_name"], 1))
    y = np.array(data[config["target_name"]])

    # split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=1)
    if (int(sys.arg[1]) == 0):
        print("Applying Linear Regression")
        # Use Linear Regression to fit the data
        model_name = os.path.join(config["model_directory"], 
                                  config["data_name"][0])
        pickled_model = pickle.load(open(model_name, 'rb'))
        run_model(X_test, y_test, pickled_model)
    else:
        print("Invalid")
        
if __name__ == 'main':
    main() 

