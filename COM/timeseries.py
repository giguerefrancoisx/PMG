# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:29:28 2018
Test the probability that a point belongs to a time series based on the probability of jumping to that point from the value at the previous time.
@author: tangk
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import medfilt

def get_diff(x,y,lb,ub,method='diff'):
    if not method in ['diff','step','alt']:
        print('Error: Method must be one of diff, step, or alt!')
        return
    if method=='step' or method=='alt':
        return np.logical_or(y[1:]-x[:-1]<lb,y[1:]-x[:-1]>ub)
    elif method=='diff':
        return np.logical_or(y-x<lb,y-x>ub)    

def mark_diff(ax,t,x,y,lb,ub,xlab=None,ylab=None,kernel_size=1,method='diff',tlim=None):
    if not method in ['diff','step','alt']:
        print('Error: Method must be one of diff, step, or alt!')
    di = np.logical_or(get_diff(x,y,lb,ub,method=method),get_diff(y,x,lb,ub,method=method))
    di = medfilt(di,kernel_size=kernel_size).astype(bool)
    if not tlim==None:
        di[tlim:] = False
    if method=='alt' or method=='step':
        ax.plot(t[:-1][di],x[:-1][di],'.r',markersize=7,alpha=0.05)
        ax.plot(t[:-1][di],y[:-1][di],'.r',markersize=7,alpha=0.05)
    elif method=='diff':
        ax.plot(t[di],x[di],'.r',markersize=7,alpha=0.05)
        ax.plot(t[di],y[di],'.r',markersize=7,alpha=0.05)
    ax.plot(t,x,'b',linewidth=1,label=xlab)
    ax.plot(t,y,'k',linewidth=1,label=ylab)   
    return ax

def get_pctile(x):
    lp = np.percentile(x,2.5)
    up = np.percentile(x,97.5)
    return lp, up  

def get_bounds(x):
    return np.min(x), np.max(x)

def get_distribution(X,i,n=1,method='diff'):
    if not method in ['diff','step','alt']:
        print('Error: Method must be one of diff, step, or alt!')
        return
    x = X[i][0::n]
    if method=='step':
        out = np.diff(x)
    elif method=='diff' or method=='alt': # alt is step but without the first differences of x
        out = np.array([])
    for j in range(len(X)):
        if j==i:
            continue
        if method=='step' or method=='alt':
            out = np.append(out,X[j][0::n][1:]-x[:-1])
        elif method=='diff':
            out = np.append(out,X[j][0::n]-x)
    return out

def estimate_ts_variance(chdata,tclist=None,channels=None,n=1,method='diff'):
    # tclist must be a dict. Keys are the categories and values are the corresponding TCs or SEs.
    if not method in ['diff','step','alt']:
        print('Error: Method must be one of diff, step, or alt!')
        return
    if tclist==None: 
        tclist = {'All':chdata.index}
    if channels==None:
        channels = chdata.columns
    lp_out = pd.DataFrame(index=tclist.keys(),columns=channels)
    up_out = pd.DataFrame(index=tclist.keys(),columns=channels)
    for ch in channels:
        for k in tclist.keys():
            dist = np.array([])
            for i, tc in enumerate(tclist[k]):
                dist = np.append(dist,get_distribution(chdata[ch][tclist[k]],i,n=n,method=method))
            if not np.isnan(dist).all():
                if np.isnan(dist).any():
                    dist = dist[~np.isnan(dist)]
                lp, up = get_pctile(dist)
                lp_out.at[k,ch] = lp
                up_out.at[k,ch] = up
            else:
                lp_out.at[k,ch] = np.nan
                up_out.at[k,ch] = np.nan
    return lp_out, up_out