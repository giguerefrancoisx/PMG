# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 12:03:29 2018

Miscellaneous helper functions

@author: tangk
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr

#%% easy numbers
def is_all_nan(data):
    """ data is in the form of an array
     return if all elements in data are np.nan"""
    return np.all(np.isnan(data))

def to_numpy(data):
    if isinstance(data,pd.core.frame.Series):
        return data.values
    elif isinstance(data,(tuple,list)):
        return np.array(data)

def r2(x,y):
    """ x and y are Series or array-like 
    Returns the R2 value between x and y """
    i = np.logical_not(np.logical_or(np.isnan(x),np.isnan(y)))
    x = x[i]
    y = y[i]
    
    lr = LinearRegression()
    x, y = to_numpy(x), to_numpy(y)
        
    lr.fit(x.reshape(-1,1),y.reshape(-1,1))
    r2 = lr.score(x.reshape(-1,1),y.reshape(-1,1))   
    return r2

def rho(x,y):
    """returns the spearman rank-order correlation coefficient"""
    i = np.logical_not(np.logical_or(np.isnan(x),np.isnan(y)))
    x = x[i]
    y = y[i]
    
    x, y = to_numpy(x), to_numpy(y)
    rho, p = spearmanr(x,y)
    return rho

def corr(x,y):
    """returns the pearson correlation coefficient"""
    i = np.logical_not(np.logical_or(np.isnan(x),np.isnan(y)))
    x = x[i]
    y = y[i]
    
    x, y = to_numpy(x), to_numpy(y)
    r, p = pearsonr(x,y)
    return r
    
#%% matching things
def ordered_intersect(l1,l2):
    """Ordered intersection of two iterables. Keeps the order of l1"""
    return [i for i in l1 if i in l2]


def arg_ordered_intersect(l1,l2):
    """Returns indices of l1 for which elements at those indices are also in l2"""
    return [i for i in range(len(l1)) if l1[i] in l2]


# fix this later
def partial_ordered_intersect(l1,l2):
    """Ordered intersection of two iterables. Keeps the order of l1. 
    If elements of l1 are not in l2, they are omitted from the returned list.
    If elements of of l2 are not in l1, they are included at the end of the 
    returned list in the order of l2."""
    l2 = list(l2)
    out = []
    for l in l1:
        if l in l2:
            out.append(l)
            l2.remove(l)
    out.extend(l2)
    return out


#%%
def condense_df(df, thresh=0.05, thresh_type='upper_bound', axis=0):
    """
    df = condense(df)
    Condenses a dataframe by removing either rows or columns where all elements
    do not meet a certain criterion. The criterion is defined by thresh_type: 
        upper_bound: thresh is a number. filters out all rows/columns where values are greater than upper_bound
        lower_bound: thresh is a number. filters out all rows/columns where values are lower than lower_bound
        between: thresh is a list of length 2. filters out all rows/columns where values are not between the lower and upper values in thresh.
        outside: thresh is a list of length 2. filters out all rows/columns where values are between the lower and upper values in thresh.
    axis=0 removes rows. axis=1 removes columns
    """
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if thresh_type=='upper_bound':
        drop = (df > thresh).all(axis=1-axis)
    elif thresh_type=='lower_bound':
        drop = (df < thresh).all(axis=1-axis)
    elif thresh_type=='between':
        drop = ((df < min(thresh)) | (df > max(thresh))).all(axis=1-axis)
    elif thresh_type=='outside':
        drop = ((df > min(thresh)) & (df < max(thresh))).all(axis=1-axis)
    drop = drop[drop].index
    return df.drop(drop, axis=axis)

# implemented in dfxtend table
#def query_list(dataframe,column,qlist):
#    """ queries a dataframe by whether column contains qlist """
#    return dataframe[dataframe[column].isin(qlist)]

# implemented in dfxtend chdata
#def re_cutoff(chdata,cutoff):
#    """ cut off the time series at a different place
#    the re cutoff must be shorter than the chdata """
#    return chdata.applymap(lambda x: x[cutoff])
    