# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:58:07 2018

@author: tangk
"""
import pandas as pd
from PMG.COM.openbook import openHDF5
from PMG.COM import arrange
import numpy as np
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
        
def initialize(directory,channels,cutoff,cat_names=None,query=None,filt=None):
    # assumes all channels are already read to HDF5
    table = read_table(directory + 'Table.csv')
    if not query==None:
        table = table.query(query)
    if not filt==None:
        table = table.filter(items=filt)
    
    t, fulldata = openHDF5(directory+'Data\\',channels=np.unique(channels))
    chdata = arrange.test_ch_from_chdict(fulldata,cutoff).filter(items=table.index,axis=0)
    t = t.get_values()[cutoff]
    
    if not cat_names==None:
        se_names = arrange.names_to_se(table,cat_names)
    else:
        se_names = None
    
    return table, t, chdata, se_names