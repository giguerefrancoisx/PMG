# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 12:03:29 2018

Miscellaneous helper functions

@author: tangk
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def is_all_nan(data):
    # data is in the form of an array
    # checks if all entries in data are np.nan
    return np.all(np.isnan(data))

def r2(x,y):
    lr = LinearRegression()
    lr.fit(x.values.reshape(-1,1),y.values.reshape(-1,1))
    r2 = lr.score(x.values.reshape(-1,1),y.values.reshape(-1,1))   
    return r2

def query_list(dataframe,column,qlist):
    # queries a dataframe by whether column contains qlist 
    return dataframe[dataframe[column].isin(qlist)]
    