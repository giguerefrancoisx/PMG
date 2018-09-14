# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:58:07 2018

@author: tangk
"""
import pandas as pd
from PMG.COM.openbook import openHDF5
from PMG.COM import arrange
import numpy as np
import os
# reads data and returns dictionary 

# as a single dictionary
# filename is a list of strings containing file names within a directory
def read_merged(directory,filename):
    full_data = {}
    for file in filename:
        print('Reading file ' + directory + file + '(SAI).csv')
        full_data[file] = pd.read_csv(directory + file + '(SAI).csv')
    return full_data

# as two dictionaries: target and bullet
# tarfile is a list of strings containing file names of targets within a directory
# bulfile is a list of strings containing file names of bullets within a directory
def read_split(directory,tarfile,bulfile):
    tar = {}
    bul = {}
    for i in len(tarfile):
        tar[tarfile[i-1]] = pd.read_csv(directory + tarfile[i-1] + '(SAI).csv')
        bul[bulfile[i-1]] = pd.read_csv(directory + bulfile[i-1] + '(SAI).csv')
    return tar, bul # return target, bullet files respectively. 

def read_table(file):
    table = pd.read_csv(file)
    if 'SE' in table.columns:
        table = table.set_index('SE',drop=True)
    elif 'TC' in table.columns:
        table = table.set_index('TC',drop=True)
    return table

def read_from_common_store(tc=None,channels=None):
    directory = 'P:\\Data Analysis\\Data\\'
    fulldata = {}
    #if either of the inputs are none, read everything
    if tc==None:
        tc = np.concatenate(pd.read_csv(directory + 'test_names.csv',header=None).values)
    tc_hdf = ['/' + i.replace('-','N') for i in tc]
    if channels==None:
        channels = np.concatenate(pd.read_csv(directory + 'test_names.csv',header=None).values)       
    with pd.HDFStore(directory + 'Tests.h5',mode='r+') as test_store:
        test_fulldata = {i: test_store.select(i) for i in tc_hdf}
    for ch in channels:
        fulldata[ch] = pd.concat([test_fulldata[i][ch].rename(i[1:].replace('N','-')) for i in tc_hdf if ch in test_fulldata[i].columns],axis=1)
    t = test_fulldata[tc_hdf[0]].iloc[:4100,0].round(4)
    return t, fulldata

def initialize(directory,channels,cutoff,cat_names=None,query=None,filt=None):
    # assumes all channels are already read to HDF5
    table = read_table(directory + 'Table.csv')
    if not query==None:
        table = table.query(query)
    if not filt==None:
        table = table.filter(items=filt)
    
    if 'Data' in os.listdir(directory):
        t, fulldata = openHDF5(directory+'Data\\',channels=np.unique(channels))
    else: # get data from P:/Data Analysis/Data
        t, fulldata = read_from_common_store(tc=table.index.values.tolist(),channels=channels)
    chdata = arrange.test_ch_from_chdict(fulldata,cutoff).filter(items=table.index,axis=0)
    t = t.get_values()[cutoff]
        
    
    if not cat_names==None:
        se_names = arrange.names_to_se(table,cat_names)
    else:
        se_names = None
    
    return table, t, chdata, se_names

def get_test(tc,channel):
    # this function is only for retrieving tests stored in P:/Data Analysis/Data
    # for now only has functionality to read one tc and one channel at a time. 
    # returns time and data from one channel and one tc
    tc = '/' + tc.replace('-','N')
    
    with pd.HDFStore('P:\\Data Analysis\\Data\\Tests.h5', mode='r+') as test:
        if channel in test[tc].columns:
            t = test[tc]['T_10000_0']
            x = test[tc][channel]
        else:
            t = None
            x = None 
    
    return t, x

def update_test_info(update_tests=True,update_channels=False):
    directory = 'P:\\Data Analysis\\Data\\'
    with pd.HDFStore(directory + 'Tests.h5') as test_store:
        tests = test_store.keys()
        if update_tests:
            test_names = [i.strip('/').replace('N','-') for i in tests]
            pd.Series(test_names).to_csv(directory + 'test_names.csv',index=False)
#            test_store.put('testNames',test_names)
        if update_channels:
            channels = []
            for t in tests:
                for ch in test_store[t].columns:
                    if not ch in channels:
                        channels.append(ch)
            pd.Series(channels).to_csv(directory + 'channel_names.csv',index=False)
#            test_store.put('channelNames',channels)
#            pd.to_csv('channel_names.csv')

def get_test_info():
    directory = 'P:\\Data Analysis\\Data\\'
    test_names = pd.read_csv(directory + 'test_names.csv',header=None)
    channel_names = pd.read_csv(directory + 'channel_names.csv',header=None)
#    with pd.HDFStore(directory + 'Tests.h5') as test_store:
#        test_names = test_store['testNames']
#        channel_names = test_store['channelNames']
    return np.concatenate(test_names.values), np.concatenate(channel_names.values)
