# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:58:07 2018

@author: tangk
"""
import pandas as pd
import numpy as np
import os
import h5py
from PMG.COM import arrange
from PMG.COM.dfxtend import * 
from PMG.COM.helper import ordered_intersect, condense_df
import json
# reads data and returns dictionary 

def read_table(file, header=[0, 1], index_col=0):
    """Reads a csv file and uses SE or TC column as the index"""
    table = pd.read_csv(file, header=header, index_col=index_col)
    
    if 'ID' in table.columns:
        table = table.set_index('ID',drop=True)
    elif 'SE' in table.columns:
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
    
    read_tc = np.unique([i for i in tc if i[:2]=='TC'])
    read_se = np.unique([i for i in tc if i[:2]=='SE'])
    
    if len(read_tc)>0:
        directories.append('P:\\Data Analysis\\Data\\TC\\')
    if len(read_se)>0:
        directories.append('P:\\Data Analysis\\Data\\SE\\')
    
    for directory in directories:
        read = read_tc if directory.endswith('TC\\') else read_se
        with h5py.File(directory+'Tests.h5','r') as test_store:
            for i in read:
                if verbose:
                    print('Retrieving ' + i)
                header = test_store[i.replace('-','N')].dtype.names
                colnames = tuple(ordered_intersect(map(lambda x: 'X'+x, channels), header))
                grp_name = i.replace('-','N')
                
                if 'XT_10000_0' in header:
                    t_i = test_store[i.replace('-','N')]['XT_10000_0']
                    start_row = np.where(np.round(t_i, 4)==-0.01)[0]
                else:
                    print('Warning: no time channel found for {}.'.format(i))
                    start_row = [0]
                    
                    
                if len(start_row)==1:
                    start_row = start_row[0]
                else:
                    print('Time channel error ({})'.format(i))
                    start_row = 0
    
                if len(colnames)==1:
                    tc_fulldata[i] = pd.DataFrame(test_store[i.replace('-','N')][colnames], columns=[colnames[0][1:]])
                else:
                    tc_fulldata[i] = arrange.h5py_to_df(test_store[i.replace('-','N')][colnames]).rename(lambda x: x.lstrip('X'),axis=1)
                tc_fulldata[i] = tc_fulldata[i].loc[start_row:].reset_index(drop=True)
    for ch in channels:
        ch_fulldata[ch] = pd.DataFrame.from_dict({i: tc_fulldata[i][ch] for i in tc if ch in tc_fulldata[i]}) 
    t = np.arange(-0.01, 0.3999, 0.0001).round(4)
#    t = tc_fulldata[tc[0]].loc[:4100,'T_10000_0'].round(4)
    return t, ch_fulldata


def filter_table(table, drop=None, query=None, multi_index_query=None, query_list=None, filt=None):
    """filters the table. """
    if drop is not None:
        table = table.drop(drop, axis=0)
    if multi_index_query is not None:
        for q in multi_index_query:
            # q is a list-like of [cols, query]
            table = table.table.multi_index_query(q[0], q[1])
    if query is not None:
        table = table.query(query)
    if query_list is not None:
        for q in query_list:
            table = table.table.query_list(q[0], q[1])
    if filt is not None:
        table = table.filter(items=filt)
    return table


def get_test(tc,channel):
    """retrieve time and data from one channel of one tc from P:\Data Analysis\Data"""
    tc = '/' + tc.replace('-','N')
    channel = 'X' + channel

    with h5py.File('P:\\Data Analysis\\Data\\' + tc[1:3] + '\\Tests.h5','r') as test:
        if 'XT_10000_0' in test[tc].dtype.names:
            t = test[tc]['XT_10000_0']
        else:
            t = np.arange(-0.01, 0.3999, 0.0001).round(4)
        
        if channel in test[tc].dtype.names:
            x = test[tc][channel]
        else:
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


#%% Classes to store data 
class Stats(object):
    """object to handle reading and processing of stats"""
    def __init__(self, directory):
        self.directory = directory
        
    def __str__(self):
        return 'Stats from directory {}'.format(self.directory)
        
    def read_json_file(self):
        """
        params = read_json_file(directory)
        Reads the data stored in params.json
        """        
        with open(self.directory + 'params.json','r') as json_file:
            params = json.load(json_file)
        self.params = params
        return self
    
    def get_test_results(self, results=['p_val']):
        """
        test_results = self.get_test_info(params)
        Returns a dict of {name: df of results}
        
        params: data stored in a serializable json format
        results: list of desired results (e.g. p-values)
        """        
        params = self.params
        test_results = {}
        for test in params['test']:
            name = test['name'] if isinstance(test['name'], str) else test['name'][0]
            test_results[name] = pd.DataFrame({res: test[res] for res in results}).applymap(np.squeeze)
        return test_results

    def summarize(self, col='p_val', condense=True, **kwargs):
        """
        df = self.summarize_test_results(col='p_val', condense=True, **kwargs)
        Gets test results for one column and returns it in a df of (rows: channels, cols: test names)
        Optionally condenses the df with kwargs
        """
        test_results = self.get_test_results(results=[col])
        summary = pd.DataFrame({name: data[col] for name, data in test_results.items()})
        if condense:
            summary = condense_df(summary, **kwargs)
        return summary
    
    
    
    
class PMGDataset(object):
    def __init__(self, directory, channels=[], cutoff=range(100, 1600), verbose=False):
        self.directory = directory
        self.channels = channels
        self.cutoff = cutoff
        self.verbose = verbose
        self.table_filters = {}
        self.preprocessing = None
        
    @property
    def table_filters(self):
        # table filters should be a dict with optional keys drop, query, query_list, and filt
        return self._table_filters
    @table_filters.setter
    def table_filters(self, val):
        self._table_filters = val
    
    @property
    def preprocessing(self):
        # preprocessing steps should be either a list of the steps or a function
        # with chdata as its only input
        return self._preprocessing
    @preprocessing.setter
    def preprocessing(self, val):
        self._preprocessing = val
    
    def read_table(self):
        table = read_table(self.directory + 'Table.csv')
        self.table = table
        return self

    def read_timeseries(self):
        t, fulldata = read_from_common_store(tc=self.table.index.tolist(), channels=self.channels, verbose=self.verbose)
        self.timeseries = arrange.to_chdata(fulldata, self.cutoff)
        self.t = t[self.cutoff]
        return self

    def read_features(self):
        features = pd.read_csv(self.directory + 'features.csv', index_col=0)
        self.features = features
    
    def read_stats(self, results=['p_val']):
        stat_obj = Stats(self.directory).read_json_file().get_test_results(results=results)
        self.stats = stat_obj

    def filter_table(self):
        kwargs = self.table_filters
        table = self.table
        table = filter_table(table, **kwargs)
        self.table = table
        return table
        
    def preprocess(self):
        if callable(self.preprocessing):
            self.timeseries = self.preprocessing(self.timeseries)
        return self.t, self.timeseries
    
    def get_data(self, datatypes):
        """retrieves the datatypes specified with preprocessing
        datatypes is a list, which can contain: 'timeseries','features','stats'.
        If datatypes is empty, only the table is read."""
            
        if 'stats' in datatypes:
            self.read_stats()
        if 'features' in datatypes:
            self.read_features()
                
        self = self.read_table()
        if len(self.table_filters)>0:
            table = self.filter_table()
        elif 'features' in datatypes:
            self.table = self.table.loc[self.features.index]
            
        if 'timeseries' in datatypes:
            self = self.read_timeseries()
            if not self.preprocessing is None:
                t, timeseries = self.preprocess()

      
# slowly phase this one out   
def initialize(directory,channels,cutoff,tc=[],verbose=False, **kwargs):
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
        table = filter_table(table, **kwargs)
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
        print('The following TCs are missing from common store:')
        print([i for i in tc if i not in common_store_tcs])
        
    chdata = arrange.to_chdata(fulldata,cutoff)
    t = t[cutoff]
        
    return table, t, chdata