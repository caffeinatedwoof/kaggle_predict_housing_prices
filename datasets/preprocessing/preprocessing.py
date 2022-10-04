# -*- coding: utf-8 -*-
"""

This script performs preprocessing on the dataset:
    
    1) Data cleansing (eliminating bad data)
    2) Data transformation 
        - encoding categorical variables
        - normalization using robust scaling
    3) Data validation (splitting data into training set and holdout set)
    
    Design decisions on data-preprocessing are documented in the accompanying 
    jupyter notebook on exploratory data analysis. 
    
"""

import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def preprocess():
    '''Performs preprocessing on dataset
    
    Returns
    -------
    tuple
        a tuple containing X_train, X_test, y_train, y_test

    '''
    #path to the dataset
    filename = "../datasets/train.csv"
    
    #Read CSV File
    train = pd.read_csv(filename, index_col='Id')
    
    #Convert YearBuilt to age
    train['age'] = 2010 - train['YearBuilt']
    
    #remove row with missing value in 'Electrical'
    train = train.drop([train[train['Electrical'].isnull()].index.values.astype(int)[0]])
    #remove other columns with missing data
    train = train.drop(train.columns[train.isnull().any()].values.astype(str),axis=1)
    #remove outlier
    train = train.drop([1299], axis=0)
    
    #Using Robust Scaling
    X = train[['OverallQual','age', 'GrLivArea', 'TotalBsmtSF']]
    transformer = RobustScaler().fit(X)
    X[['OverallQual','age', 'GrLivArea', 'TotalBsmtSF']] = transformer.transform(X)
    X = pd.concat((X, train[['Neighborhood', 'SaleCondition']]), axis=1)
    
    #Creating dummy columns using one-hot encoding
    def create_dummies(df,column_name):
        dummies = pd.get_dummies(df[column_name],prefix=column_name)
        df = pd.concat([df,dummies],axis=1)
        return df
    
    X = create_dummies(X, "Neighborhood")
    X = create_dummies(X, "SaleCondition")
    X = X.drop(labels=['Neighborhood', 'SaleCondition'],axis=1)


    #Using log transform on Sale Price
    y = np.log(train['SalePrice'])
    
    #Split data into training set and holdout set in 80/20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    return X_train, X_test, y_train, y_test

def main():
    preprocess()

if __name__ == '__main__':
    main()
    