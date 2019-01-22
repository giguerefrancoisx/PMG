# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:58:07 2018

@author: tangk
"""
import pandas as pd
import numpy as np
import os
import h5py
from PMG.COM.openbook import openHDF5
from PMG.COM import arrange
from PMG.COM.dfxtend import * 
from PMG.COM.helper import ordered_intersect
# reads data and returns dictionary 



def read_table(file):
    """Reads a csv file and uses SE or TC column as the index"""
    table = pd.read_csv(file)
    if 'SE' in table.columns:
        table = table.set_index('SE',drop=True)
    elif 'TC' in table.columns:
        table = table.set_index('TC',drop=True)
    return table


def read_from_common_store(tc,channels,verbose=False):
    """Reads TCs and SEs from the P:\Data Analysis\Data directory
    tc can be a mix of TCs and SEs"""
    directories = []
    ch_fulldata = dict.fromkeys(channels)
    tc_fulldata = dict.fromkeys(tc)
    
    read_tc = [i for i in tc if i[:2]=='TC']
    read_se = [i for i in tc if i[:2]=='SE']
    
    if len(read_tc)>0:
        directories.append('P:\\Data Analysis\\Data\\TC\\')
    if len(read_se)>0:
        directories.append('P:\\Data Analysis\\Data\\SE\\')
    
    for directory in directories:
        with h5py.File(directory+'Tests.h5','r') as test_store:
            for i in tc:
                if verbose:
                    print('Retrieving ' + i)
                header = test_store[i.replace('-','N')].dtype.names
                colnames = tuple(ordered_intersect(map(lambda x: 'X'+x, channels), header))
                if 'XT_10000_0' not in colnames:
                    colnames = tuple(['XT_10000_0']) + colnames
                tc_fulldata[i] = arrange.h5py_to_df(test_store[i.replace('-','N')][colnames]).rename(lambda x: x.lstrip('X'),axis=1)
        for ch in channels:
            ch_fulldata[ch] = pd.DataFrame.from_dict({i: tc_fulldata[i][ch] for i in tc if ch in tc_fulldata[i]})                
    t = tc_fulldata[tc[0]].loc[:4100,'T_10000_0'].round(4)
    return t, ch_fulldata


def initialize(directory,channels,cutoff,tc=[],query=None,query_list=[],filt=None,drop=None,verbose=False):
    """retrieve data and Table.csv"""
    # check channels
    if not isinstance(channels,(list,np.ndarray,tuple)):
        print('Channels in the wrong format!')
        return
    elif len(channels)==0:
        print('No channels specified!')
        return
    else:
        channels = np.unique(channels)
    
    # check TCs and get table
    if 'Table.csv' in os.listdir(directory):
        # read table, get TCs
        table = read_table(directory + 'Table.csv')
        if not drop==None:
            table = table.drop(drop,axis=0)
        if not query==None:
            table = table.query(query)
        if not filt==None:
            table = table.filter(items=filt)
        if len(query_list)>0:
            for q in query_list:
                table = table.table.query_list(q[0],q[1])
        tc = table.index.values.tolist()
        

    elif len(tc)>0:
        table = pd.DataFrame([])
    else:
        print('No TCs specified!')
        return

    # read data from common store or Data directory if not available in common store
    common_store_tcs = np.squeeze(pd.read_csv('P:\\Data Analysis\\Data\\test_names.csv',header=None).values)
    in_common_store = [i in common_store_tcs for i in tc]
    if all(in_common_store):
        t, fulldata = read_from_common_store(tc=tc, channels=channels, verbose=verbose)
    else:
        t, fulldata = openHDF5(directory + 'Data\\', channels = channels)
        
    chdata = arrange.to_chdata(fulldata,cutoff).filter(items=tc,axis=0)
    t = t.get_values()[cutoff]
    chdata.chdata.t = t
        
    return table, t, chdata


def get_test(tc,channel):
    """retrieve time and data from one channel of one tc from P:\Data Analysis\Data"""
    tc = '/' + tc.replace('-','N')
    channel = 'X' + channel

    with h5py.File('P:\\Data Analysis\\Data\\' + tc[1:3] + '\\Tests.h5','r') as test:
        if channel in test[tc].dtype.names:
            t = test[tc]['XT_10000_0']
            x = test[tc][channel]
        else:
            t = None
            x = None 
    
    return t, x


def get_se_angle(se):
    """ reads from P:\Data Analysis\Data\angle_213 which stores data of tracked targets on RFCS. 
    returns a dataframe similar to chdata, i.e. of shape (n_tests,5) where the columns are
    Time, Up_x, Up_y, Down_x, Down_y"""
    directory = 'P:\\Data Analysis\\Data\\angle_213\\'
    with h5py.File(directory + 'Tests.h5','r') as test:
        se = [s for s in map(lambda x: x.replace('-','N'),se) if s in test.keys()]
        se_list = [arrange.h5py_to_df(test[s]).apply(tuple,axis=0).apply(np.array) for s in se]
    df = pd.concat(se_list,axis=1,ignore_index=True).T
    df.index = [s.replace('N','-') for s in se]
    return df
            

def update_test_info(update_tests=True,update_channels=True):
    """Update test_names.csv and/or channel_names.csv"""
    directory = 'P:\\Data Analysis\\Data\\'
    if update_tests:
        test_names = []
    if update_channels:
        channels = []
        
    for sub in ['TC','SE']:
        with h5py.File(directory+sub+'\\Tests.h5','r') as test_store:
            tests = list(test_store.keys())
            if update_tests:
                test_names.extend([i.strip('/').replace('N','-') for i in tests])
            if update_channels:
                for t in tests:
                    for ch in [i.lstrip('X') for i in test_store[t].dtype.names]:
                        if not ch in channels:
                            channels.append(ch)
    if update_tests:
        pd.Series(test_names).to_csv(directory + 'test_names.csv',index=False)
    if update_channels:
        pd.Series(channels).to_csv(directory + 'channel_names.csv',index=False)


def get_test_info():
    """returns list of all tcs/ses and channels in HDF5"""
    directory = 'P:\\Data Analysis\\Data\\'
    test_names = pd.read_csv(directory + 'test_names.csv',header=None)
    channel_names = pd.read_csv(directory + 'channel_names.csv',header=None)
    return np.concatenate(test_names.values), np.concatenate(channel_names.values)

#--------get rid of these eventually------------

## old code
#def read_from_common_store(tc,channels,verbose=False):
#    """Read from the HDF5 file that stores all of the tests"""
#    directories = []
#    fulldata = {}
#    
#    read_tc = [i for i in tc if i[:2]=='TC']
#    read_se = [i for i in tc if i[:2]=='SE']
#    if len(read_tc)>0:
#        directories.append('P:\\Data Analysis\\Data\\TC\\')
#    if len(read_se)>0:
#        directories.append('P:\\Data Analysis\\Data\\SE\\')
#
#    for directory in directories: 
#        with h5py.File(directory+'Tests.h5','r') as test_store:
#            test_fulldata = {i: arrange.h5py_to_df(test_store[i.replace('-','N')]).rename(lambda x: x.lstrip('X'),axis=1) for i in tc}
#        for ch in channels:
#            if not ch in fulldata:
#                fulldata[ch] = pd.concat([test_fulldata[i][ch].rename(i) for i in tc if ch in test_fulldata[i].columns],axis=1)
#            else:
#                concat_to_df = pd.concat([test_fulldata[i][ch].rename(i) for i in tc if ch in test_fulldata[i].columns],axis=1)
#                fulldata[ch] = pd.concat([fulldata[ch],concat_to_df],axis=1)
#        t = test_fulldata[tc[0]].iloc[:4100,0].round(4)
#    return t, fulldata
    

# OBSOLETE
# as a single dictionary
# filename is a list of strings containing file names within a directory
#def read_merged(directory,filename):
#    full_data = {}
#    for file in filename:
#        print('Reading file ' + directory + file + '(SAI).csv')
#        full_data[file] = pd.read_csv(directory + file + '(SAI).csv')
#    return full_data