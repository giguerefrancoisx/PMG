# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:25:33 2018
Functions related to rearranging data
@author: tangk
"""
import pandas as pd
import numpy as np
from PMG.COM.helper import *

#%% rearranging things     
def to_chdata(data, cutoff=None):
    """ returns data in the format of chdata
    chdata is a dataframe of size (n_tests, n_channels)
    each element is an array-type of the time series or is nan
    data are cut to the cutoff
    data is a dict of either {tc: df_by_ch} or {ch: df_by_tc} 
    """
    
    # initialize: columns are the keys of dict
    chdata = pd.DataFrame(index=get_unique(data), columns=data.keys())
    if cutoff:
        nan_list = [np.tile(np.nan,len(cutoff))]
    
    # populate each column with data
    for col in chdata.columns:
        if not cutoff:
            cutoff = range(data[col].shape[0])
            nan_list = [np.tile(np.nan,len(cutoff))]
        # if there is only one value, put it at the beginning of the range so it won't be cut off
        if data[col].replace(0,np.nan).dropna(axis=0,how='all').shape[0]==1: 
            data[col].loc[cutoff[0]] = data[col].loc[0]
        chdata.loc[data[col].columns, col] = data[col].apply(lambda x: tuple(x.loc[cutoff])).apply(np.array)
        chdata.loc[chdata.index.drop(data[col].columns), col] = nan_list
        
    if all(['TC' in i or 'SE' in i for i in chdata.columns]):
        return chdata.T
    else:
        return chdata


def h5py_to_df(data):
    """convert a compound dataset written using h5py into a pandas dataframe"""
    return pd.DataFrame(data[:],columns=data.dtype.names)


def arrange_by_group(table,data,dict_key,col_key=None,col_order=[]):
    """rearranges the data to a dict of {category: pd.DataFrame or pd.Series}.
    Categories are determined by the unique values under column dict_key of table. 
    pd.DataFrame columns are determined by the unique values under column col_key of table.
    Data is a pd.Series. col_order specifies the order of the pd.DataFrame columns."""
    
    # remove tests with missing values
    table = table.loc[~data.loc[table.index].apply(is_all_nan)]
    grouped = table.groupby(dict_key)
    
    if col_key==None:
        groups = grouped.groups
        return {grp: data.loc[groups[grp]] for grp in groups}
    else:
        x = {}
        for grp in grouped:
            subgroups = grp[1].groupby(col_key).groups
            x[grp[0]] = pd.DataFrame.from_dict({sg: data.loc[subgroups[sg]] for sg in subgroups})
            if not col_order==[]:
                col_order = partial_ordered_intersect(col_order,x[grp[0]].columns)
                x[grp[0]] = x[grp[0]][col_order]
        return x


def unpack(data):
    """unpacks Series of arrays into DataFrame with size (n_timepoints, n_samples)"""
    return pd.DataFrame(np.vstack(data.values),index=data.index).T
#%% reindexing things
def align(t1,t2,data):
    """ t1 and t2 are both arrays of times
    data is a timeseries whose time corresponds to t2
    returns an array aligned that has the same length as t1 and contains the 
    elements of data at times that are observed in both t1 and t2
    
    e.g. t1 = [1,2,3,4,5]
         t2 = [2,4,5]
         data = [10,20,30]
         align(t1,t2,data)
         >>> [nan, 10, nan, 20, 30]"""
    t1, t2, data = np.asarray(t1), np.asarray(t2), np.asarray(data)
    
    aligned = np.tile(np.nan,len(t1))
    for i, t in enumerate(t1):
        if t in t2:
            aligned[i] = data[np.squeeze(np.where(t2==t))]
    return aligned

#%% matching things
def get_unique(data):
    """ data is a dict of dataframes, e.g. {tc: df_by_ch} or {ch: df_by_tc}
    returns the unique list of columns in the dataframes"""
    full_list = []
    for i in data.keys():
        full_list.extend(data[i].columns)
    return list(dict.fromkeys(full_list))


def intersect_columns(x):
    """x is a dict of {category: pd.DataFrame}.
    Checks that the columns of all the dataframes are the same. 
    Returns a new dict with matched columns."""
    columns = [x[k].columns for k in x]
    if len(columns)==0:
        return
    keep_columns = columns[0]
    for col in columns[1:]:
        keep_columns = ordered_intersect(keep_columns,col)
    return {k: x[k][keep_columns] for k in x}


# fix this    
def match_groups(x, y):
    """x and y are two outputs of arrange_by_group. This function checks 
    that x and y have the same number of elements in each key:value pair"""
    rm_keys = []
    for k in x:
        i = ordered_intersect(x[k].index,y[k].index)
        if i==[]:
            rm_keys.append(k)
        else:
            x[k] = x[k].loc[i]
            y[k] = y[k].loc[i]
    while len(rm_keys)>0:
        k = rm_keys.pop()
        _ = x.pop(k)
        if k in y:
            _ = y.pop(k)


def match_keys(x, y):
    keys = ordered_intersect(list(x.keys()), list(y.keys()))
    x = {k: x[k] for k in keys}
    y = {k: y[k] for k in keys}
    return x, y
    
#--------------- get rid of these eventually ----------------------------------    
## data is a dataframe
#def arrange_by_peak(data):
#    nkey = len(data.keys())
#    chkey = data.keys()
#    for i in range(nkey,0,-1):
#        chkey = chkey.insert(i,data.keys()[i-1])
#
#    peakkey = np.matlib.repmat(np.array(['-tive','+tive']),1,nkey).flatten()
#    
#    out = pd.DataFrame([],columns=[chkey,peakkey])
#    
#    for ch in data.keys():
#        for i in data.index:
#            out.set_value(i,(ch,'-tive'),data.get_value(i,ch)[0])
#            out.set_value(i,(ch,'+tive'),data.get_value(i,ch)[1])
#    return out

#def sep_by_peak(data):
#    return data.applymap(get_min), data.applymap(get_max)

#def get_min(data):
#    return data[0]

#def get_max(data):
#    return data[1]

#def get_values(data):
#    return data[~np.isnan(data)]
            