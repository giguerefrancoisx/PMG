# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:04:32 2019

Functions that I don't want in the main script 

@author: tangk
"""
import pandas as pd
import numpy as np
from PMG.COM.helper import *
from PMG.COM.arrange import unpack
from scipy.signal import correlate


#%% distance metrics
def get_lag(x, y):
    """gets the lag between x and y. x and y are both pd.DataFrames with columns
    ['DUp_x','DUp_y','DDown_x','DDown_y']"""
    lag = []
    for ch in x.columns:
        xcorr = correlate(x[ch].values, y[ch].values)
        lag.append(np.argmax(xcorr)-len(xcorr)//2)
    lag = int(np.mean(lag))
    print(lag)
    return lag


def get_lag2(x,y):
    """gets the lag between x and y by minimizing the error between them"""
    lag = []
    max_lag = len(x)//2
    for ch in x.columns:
        errs = pd.Series({lag: error_function(lag,x[ch].values,y[ch].values) for lag in range(-max_lag,max_lag+1)})
        lag.append(errs.sort_values().index[0])
    lag = int(np.mean(lag))
    print(lag)
    return lag
     
        
def error_function(lag, x, y):
    lag = int(lag)
    if lag<0:
        x = x[:-(-lag)]
        y = y[(-lag):]
    elif lag>0:
        x = x[lag:]
        y = y[:-lag]
    return np.sqrt((x-y)**2).sum()

        
def get_aligned(x,y,lagfun=get_lag2):
    """aligns x and y
    x and y are both pd.Series with columns ['DUp_x','DUp_y','DDown_x','DDown_y']
    returns x and y with the same structure as the input
    lagfun is the function used to calculate the lag"""
    x = unpack(x)
    y = unpack(y)
    
    lag = lagfun(x,y) 
#    print(lag)
    y = y.shift(lag)
    
    if lag>0:
        y = y.fillna(0)
    elif lag<0:
        y = y.dropna(axis=0)
    return y.apply(tuple).apply(np.array)


def get_distance(x,y,align=True,lagfun=get_lag2):
    """ gets distance metric between x and y. x and y are both pd.DataFrames with
    size (n_sample, 4) with columns ['DUp_x','DUp_y','DDown_x','DDown_y']. 
    Arbitrarily aligns everything to the first sample."""
    if isinstance(x,pd.core.series.Series):
        x = pd.DataFrame(x).T
    if isinstance(y,pd.core.series.Series):
        y = pd.DataFrame(y).T
    
    # get rid of nan
    x = x.loc[~x.applymap(is_all_nan).all(axis=1)]
    y = y.loc[~y.applymap(is_all_nan).all(axis=1)]
    
    if len(x)==0 or len(y)==0:
        return np.nan
    
    # cut time series to have the same length
    min_len = pd.concat((x.iloc[:,0].apply(len), y.iloc[:,0].apply(len))).min()
#    print(min_len)
    x, y = x.applymap(lambda x: x[:min_len]), y.applymap(lambda y: y[:min_len])    

    if align:
        # align time series
        for i in range(len(x)-1):
            x.iloc[i+1,:] = get_aligned(x.iloc[0,:], x.iloc[i+1,:],lagfun=lagfun)
        for i in range(len(y)):
            y.iloc[i,:] = get_aligned(x.iloc[0,:], y.iloc[i,:],lagfun=lagfun)
        
        min_len = pd.concat((x.iloc[:,0].apply(len), y.iloc[:,0].apply(len))).min()
        x, y = x.applymap(lambda x: x[:min_len]), y.applymap(lambda y: y[:min_len]) 
    
    # calculate mean of each
    x = x.apply(lambda x: np.mean(np.stack(x.values),axis=0))
    y = y.apply(lambda x: np.mean(np.stack(x.values),axis=0))
    
    # review this metric
    dist_upper = np.sqrt((((x-y)[['DUp_x','DUp_y']])**2).sum(axis=1)).mean()
    dist_lower = np.sqrt((((x-y)[['DDown_x','DDown_y']])**2).sum(axis=1)).mean()
    dist = dist_upper + dist_lower
    
#    dist = (np.sqrt(((x-y)**2).sum(axis=0))/min_len).sum()
    return dist

def get_var(x):
    """x is a pd.DataFrame
    gets std with the restriction that mean-std cannot be less than zero"""
    means = x.mean()
    err_upper = x.std()
    err_lower = x.std()
    i = err_lower>means
    err_lower[i] = means[i]
#    err_lower[err_lower>means] = 0
    
    return pd.concat((err_lower,err_upper),axis=1).apply(tuple)