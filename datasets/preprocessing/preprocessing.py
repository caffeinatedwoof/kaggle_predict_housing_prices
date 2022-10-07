# -*- coding: utf-8 -*-
"""

This script performs preprocessing on the dataset:
    
    1) Data cleansing (eliminating bad data)
    2) Data transformation 
        - feature engineering 
        - encoding categorical variables
        - normalization using robust scaling
    3) Data validation (splitting data into training set and holdout set)
    
    Design decisions on data-preprocessing are documented in the accompanying 
    jupyter notebook on exploratory data analysis. 
    
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../datasets', 'preprocessing'))


def preprocess(train):
    '''Performs preprocessing on dataset
    
    Returns
    -------
    tuple
        a tuple containing X_train, X_test, y_train, y_test

    '''
    
    #remove row with missing value in 'Electrical'
    train = train.drop([train[train['Electrical'].isnull()].index.values.astype(int)[0]])
    #remove other columns with missing data
    train = train.drop(train.columns[train.isnull().any()].values.astype(str),axis=1)
    #remove outlier
    train = train.drop([1299], axis=0)

    #combine TotalBsmtSF & GrLivArea
    train['TotalSF'] = train['GrLivArea'] + train['TotalBsmtSF']
    
    #binning neighborhoods
    all_nbhds = list(train['Neighborhood'].unique())
    neighborhoods = {}
    for nbhd in all_nbhds:
        if nbhd in ['MeadowV', 'IDOTRR', 'BrDale']:
            neighborhoods[nbhd] = 1
        elif nbhd in ['StoneBr','NridgHt', 'NoRidge']:
            neighborhoods[nbhd] = 4
        elif nbhd in ['BrkSide', 'Edwards', 'OldTown', 'Sawyer', 'Blueste', 
                      'SWISU', 'NPkVill', 'NAmes', 'Mitchel']:
            neighborhoods[nbhd] = 2
        else:
            neighborhoods[nbhd] = 3
        
    train['Neighborhood'] = train['Neighborhood'].apply(lambda x: neighborhoods[x]) 
    
    #Adding Age, Remodelled and IsNew
    train['Age'] = train['YrSold'] - train['YearRemodAdd']
    train['Remodelled'] = train.apply(lambda x: 0 if 
                                      (x['YearBuilt'] == x['YearRemodAdd']) 
                                      else 1, axis=1)
    train['IsNew'] = train.apply(lambda x: 1 
                                 if (x['YrSold'] == x['YearBuilt']) else 0, axis=1)
    
    #Converting non-numeric predictors to cateogrical variables
    train['YrSold'] = pd.Categorical(train['YrSold'], ordered=True)
    train['MoSold'] = pd.Categorical(train['MoSold'], ordered=False)
    train['OverallQual'] = pd.Categorical(train['OverallQual'], ordered=True)
    train['Neighborhood'] = pd.Categorical(train['Neighborhood'], ordered=True)
    
    #Using Robust Scaling
    X = train[['Age', 'TotalSF']]
    transformer = RobustScaler().fit(X)
    X[['age', 'TotalSF']] = transformer.transform(X)
    X = pd.concat((X, train[['SaleCondition','Neighborhood','OverallQual','Age', 
                             'TotalSF', 'Age', 'Remodelled', 'IsNew', 'YrSold', 
                             'MoSold']]), 
                  axis=1)

    
    #Creating dummy columns using one-hot encoding
    def create_dummies(df,column_name):
        dummies = pd.get_dummies(df[column_name],prefix=column_name)
        df = pd.concat([df,dummies],axis=1)
        return df
    
    X = create_dummies(X, "SaleCondition")
    X = X.drop(labels=['SaleCondition'],axis=1)


    #Using log transform on Sale Price
    y = np.log(train['SalePrice'])
    
    #Split data into training set and holdout set in 80/20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    return X_train, X_test, y_train, y_test

def main():
    preprocess(train)

if __name__ == '__main__':
    main()
    