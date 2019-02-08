# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:29:28 2018
Test the probability that a point belongs to a time series based on the 
probability of jumping to that point from the value at the previous time.

Methods 'diff', 'step', and 'alt' refer to different methods of getting the 
samples to form an empirical distribution of steps:
    - Diff = samples are y(t)-x(t), where x and y are timeseries from replicate tests
    - Step = samples are x(t+1)-x(t) and y(t+1)-x(t), where x and y are timeseries from replicate tests
    - Alt = samples are y(t+1)-x(t), where x and y are timeseries from replicate tests

@author: tangk
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import medfilt
from PMG.COM.helper import is_all_nan

def check_method(method):
    """checks that the method used is valid"""
    assert(method in ['diff','step','alt']), 'Method must be one of diff, step, or alt!'        

def subsample(X,n):
    """subsamples every n points from X"""
    if isinstance(X,pd.core.series.Series):
        return X.apply(lambda x: x[0::n])
    else:
        raise Exception('Sub sampling not implemented for non-Series types')

def get_distribution(X,i,n=1,method='diff'):
    """gets samples used for empirical distribution
    X is a matrix of [n_timeseries, n_timepoints]
    i is the index of the reference timeseries
    step size is n"""
    if n>1:
        X = subsample(X,n)
    x = X[i]
    
    if method=='step':
        out = np.diff(x)
    else:
        out = np.array([])
        
    for j in range(len(X)):
        if j==i:
            continue
        if method in ['step','alt']:
            out = np.append(out,X[j][1:]-x[:-1])
        else:
            out = np.append(out,X[j]-x)
    return out

def get_composite_distribution(X,n=1,method='diff'):
    """does get_distribution for all tests in X"""
    dist = np.array([])
    for i in range(len(X)):
        dist = np.append(dist,get_distribution(X,i,n=n,method=method))
    return dist

def get_diff(x,y,lb,ub,method='diff'):
    """returns an array of boolean comparing at each time point whether the 
    difference between x and y (computed using one of the three methods) 
    exceeds lb or ub. Note that lb is negative. """
    if method=='step' or method=='alt':
        return np.logical_or(y[1:]-x[:-1]<lb,y[1:]-x[:-1]>ub)
    elif method=='diff':
        return np.logical_or(y-x<lb,y-x>ub)    

def get_pctile(x):
    """returns the 2.5th and 97.5th percentiles of an empirical distribution with samples given by x"""
    return np.percentile(x,2.5), np.percentile(x,97.5)

def get_bounds(x):
    """returns the min and max of samples given by x"""
    return np.min(x), np.max(x)

def compare_two(x,y,lb,ub,method='diff',kernel_size=1):
    """compare two time series x and y according to method and with bounds lb 
    and ub, then apply a median filter of size kernel_size"""
    di = np.logical_or(get_diff(x,y,lb,ub,method=method),get_diff(y,x,lb,ub,method=method))
    di = medfilt(di,kernel_size=kernel_size).astype(bool)
    return di

def estimate_ts_variance(chdata,tclist=None,channels=None,n=1,method='diff',bounds='pctile'):
    """ Estimates upper and lower bounds of how large a step size must be for 
    two timeseries to be considered significantly different from each other at 
    a point in time. 
    
    chdata is chdata
    tclist is a dict with {group name: TC within the group name}
    channels is a list of the channels to be compared
    n is the number of steps to take (i.e. changes sampling rate)
    method is the method used to get the empirical distribution of step sizes
    bounds is either 'pctile'--95th percentile bounds, or 'range'--min and max
    
    Returns lp_out and up_out, two DataFrames which respectively store the 
    lower and upper bounds for each group in tclist"""
    
    check_method(method)
    
    if tclist==None: 
        tclist = {'All':chdata.index}
    if channels==None:
        channels = chdata.columns
    
    lp_out = pd.DataFrame(index=tclist.keys(),columns=channels)
    up_out = pd.DataFrame(index=tclist.keys(),columns=channels)
    
    for ch in channels:
        for k in tclist.keys():
            X = chdata.loc[tclist[k],ch]
            X = X[~X.apply(is_all_nan)]
            if (len(X)==0) or (len(X)==1 and method!='step'):
                lp_out.at[k,ch] = np.nan
                up_out.at[k,ch] = np.nan
                continue

            dist = get_composite_distribution(X,n=n,method=method)
            if bounds=='pctile':
                lp, up = get_pctile(dist)
            elif bounds=='range':
                lp, up = get_bounds(dist)
            lp_out.at[k,ch], up_out.at[k,ch] = lp, up
    return lp_out, up_out


def mark_diff(ax,t,x,y,lb,ub,xlab=None,ylab=None,kernel_size=1,method='diff',tlim=None):
    """plots two time series x and y and marks where there are significant 
    differences between the two using bounds lb and ub
    lb is a negative number
    tlim masks the timeseries so that all time points after tlim are not considered"""
    check_method(method)
    
    di = compare_two(x,y,lb,ub,method=method,kernel_size=kernel_size)
    if tlim:
        di[tlim:] = False
    if method in ['alt','step']:
        ax.plot(t[:-1][di],x[:-1][di],'.r',markersize=7,alpha=0.05)
        ax.plot(t[:-1][di],y[:-1][di],'.r',markersize=7,alpha=0.05)
    elif method=='diff':
        ax.plot(t[di],x[di],'.r',markersize=7,alpha=0.05)
        ax.plot(t[di],y[di],'.r',markersize=7,alpha=0.05)
        
    ax.plot(t,x,'b',linewidth=1,label=xlab)
    ax.plot(t,y,'k',linewidth=1,label=ylab)   
    return ax