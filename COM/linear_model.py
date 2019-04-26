# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:09:10 2019

helper functions for linear models

@author: tangk
"""

def get_model(model, x, y, *args, **kwargs):
    """defines an instance of class model, drops missing values from x and y,
    and returns the model fit to x and y. args and kwargs are specifications
    of the model instance
    x and y are either pandas Series or dataframes"""
    model_inst = model(*args, **kwargs)
    
    if len(x.shape)==1:
        x = x.to_frame()
    if len(y.shape)==1:
        y = y.to_frame()
        
    drop = x.loc[x.isna().any(axis=1)].append(y.loc[y.isna().any(axis=1)]).index
    x = x.drop(drop)
    y = y.drop(drop)
    
    model_inst = model_inst.fit(x, y)
    return model_inst, x, y
        
