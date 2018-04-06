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
        for f in files:
            if f in data[ch].keys():
                chdata.set_value(f,ch,data[ch][f].get_values()[cutoff])
            else:
                chdata.set_value(f,ch,np.matlib.repmat(np.nan,1,len(list(cutoff))))
    return chdata
    
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
            out.set_value(i,ch,data.get_value(i,ch))
    return out

def get_values(data):
    return data[~np.isnan(data)]