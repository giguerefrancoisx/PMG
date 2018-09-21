# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:25:33 2018

@author: tangk
"""
import pandas as pd
import numpy as np

# arrange data to types
# input is always dict form (i.e. output of read_data)

def test_ch(data,channels,cutoff):
    chdata = pd.DataFrame(index=data.keys(),columns=channels)
    
    for ch in channels:
        for f in data.keys():
            chdata.set_value(f,ch,pd.DataFrame(data[f],columns=[ch]).get_values().flatten()[cutoff].tolist())
    return chdata

def test_ch_from_chdict(data,cutoff):
    channels = list(data.keys())
    
    # get names of files
    files = []
    for ch in channels:
        newfiles = data[ch].keys()
        for f in newfiles:
            if not(f in files):
                files.append(f)
    
    chdata = pd.DataFrame(index=files,columns=channels)
    
    for ch in channels:
        f = data[ch].columns
        if ((data[ch].loc[cutoff]==0) | (np.isnan(data[ch].loc[cutoff]))).all().all():
            chdata[ch][f] = data[ch].apply(lambda x: tuple([x.iloc[0]])).apply(np.array)
            for f2 in chdata.index.drop(f):
                chdata[ch][f2] = np.array([np.nan])
        else:
            chdata[ch][f] = data[ch].apply(lambda x: tuple(x.loc[cutoff])).apply(np.array)
            for f2 in chdata.index.drop(f):
                chdata[ch][f2] = np.tile(np.nan,len(list(cutoff)))
    
    return chdata

def t_ch_from_test_ch(data):
    out = {}
    for ch in data.columns:
        out[ch] = pd.DataFrame(np.vstack(data[ch].values),index=data.index).T
    return out

def names_to_se(table,names):
    # table is the table
    # names is a dict with keys categorical names and entires query parameters
    # return a dict with keys categorical names and entries the corresponding
    out = {}
    for n in names.keys():
        out[n] = list(table.query(names[n]).index)
    return out
    
# data is a dataframe
def arrange_by_peak(data):
    nkey = len(data.keys())
    chkey = data.keys()
    for i in range(nkey,0,-1):
        chkey = chkey.insert(i,data.keys()[i-1])

    peakkey = np.matlib.repmat(np.array(['-tive','+tive']),1,nkey).flatten()
    
    out = pd.DataFrame([],columns=[chkey,peakkey])
    
    for ch in data.keys():
        for i in data.index:
            out.set_value(i,(ch,'-tive'),data.get_value(i,ch)[0])
            out.set_value(i,(ch,'+tive'),data.get_value(i,ch)[1])
    return out

def sep_by_peak(data):
    return data.applymap(get_min), data.applymap(get_max)

def get_min(data):
    return data[0]

def get_max(data):
    return data[1]

def get_values(data):
    return data[~np.isnan(data)]

def h5py_to_df(data):
    #convert a compound dataset written using h5py into a pandas dataframe
    return pd.DataFrame(data[:],columns=data.dtype.names)