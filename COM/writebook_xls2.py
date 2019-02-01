# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:04:56 2018
Read HDF5 written by h5py using h5py
@author: tangk

"""
import os
import pandas
import numpy
import re
import h5py
import xlwings as xw
from xlrd import XLRDError



def match_columns(table, regex='\d{2}[A-Z0-9]{14}'):
    """table is a pandas DataFrame
    Goes through row by row until something resembling channel names is found
    Optionally, takes a custom regex. The default one looks for the default 
    coding standard for channel names. rows can either be the first row to
    start searching or a range of rows to search over"""
    
    for i in table.index:
        contains_exp = table.loc[i].str.contains(regex)
        if contains_exp.any():
            return table.loc[i, contains_exp].to_dict(), i
    return {}, -1


def get_colnames(table, channel_exp='\d{2}[A-Z0-9]{14}', time_exp='T_10000_0'):
    names, row = match_columns(table, regex=channel_exp)
    t_names, t_row = match_columns(table, regex=time_exp)
    if len(t_names)==0:
        return names, row
    elif len(t_names)>1:
        print('Error: number of time columns >1. Check regexp.')
        return
    if row!=t_row: 
        print('Error: time column and channel column were not in the same row. Check data.')
        return
    names.update(t_names)
    return names, row
    
    
def get_data(table, row_start=0, colnames={}):
    """table is a pandas DataFrame
    Returns the row and column where the data begin.
    Assumes that single value measurements e.g. hic have values 0 after index 0"""
    if len(colnames)>0:
        table = table.rename(colnames, axis=1)
    
    is_str = table.applymap(type)==str
    table[is_str] = numpy.nan
    table = table.dropna(axis=1, how='all').dropna(axis=0, how='any')
    return table


def read_table_xw(path, empty_val='NA'):
    """reads a table using xlwings"""
    book = xw.Book(path)
    table = pandas.DataFrame(book.sheets[0].used_range.options(empty=empty_val).value)
    book.close()
    return table


def read_excel(path):
    """reads an excel file (extension .xls or .xlsx) using pandas or xlwings."""
    try:
        testframe = pandas.read_excel(path,sheet_name=None, header=0,index_col=0, skiprows=[1,2],dtype=numpy.float64)
        testframe = pandas.concat(list(testframe.values()), axis=1)
        try:
            start = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], -0.01001))
            end = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], 0.3999))+1
        except KeyError as e:
            raise KeyError('{} of test: {}. File has a different arangement '\
                           'of data or is missing time column'.format(e, path))
        testframe = testframe.iloc[start:end,:]
        
    except:
        testframe = read_table_xw(path)
        colnames, row_index = get_colnames(testframe)
        if 'T_10000_0' not in colnames.values():
            print('Warning: T_10000_0 not in ' + path)
        testframe = testframe.rename(colnames, axis=1)
        testframe = get_data(testframe)
    
    return testframe
    

def check_testframe(tf):
    """checks testframe to make sure the data imported and column names 
    are OK. Returns 0 if autocheck failed and 1 if autocheck passed"""
    if isinstance(tf, pandas.core.series.Series):
        return 'series'
    if min(tf.shape)==0 or tf.isna().all().all():
        return 'empty'
    if len(tf)<2:
        return 'truncated'
    if not tf.applymap(numpy.isreal).all().all():
        return 'nonreal'
    if 'T_10000_0' in tf.columns:
        tf = tf.drop('T_10000_0', axis=1)
    ch_names = tf.filter(regex='\d{2}[A-Z0-9]{14}', axis=1).columns
    if len(ch_names) !=  len(tf.columns):
        return 'colnames' + str(tf.columns.drop(ch_names))
    return 'ok'


def check_filenames(filenames, regex='[TS][CE]\d{2}-\d{3,4}.'):
    """checks file names before reading files to make sure
    the files to read are OK. Filenames is a list-like. Returns 'ok' if 
    autocheck passed and 'unmatched names' otherwise"""
    rc = re.compile(regex)
    matched = list(filter(rc.match, filenames))
    if len(matched) != len(filenames): 
        return 'unmatched names'
    else:
        return 'ok'

# to do: add option of editing testframe and re-checking
def writeHDF5(directory, file_check=1, data_check=1):
    """reads .xls, .xlsx, and .csv data files and writes them to HDF5
    Optionally check filenames and data before reading and writing. Values of
    file_check and data_check correspond to:
        0: no check
        1: exit if autocheck fails
        2: if autocheck passes, print and continue with user input"""

    allfiles = [file for file in os.listdir(directory) if file.endswith(('.xls','.xlsx','.csv'))]
    
    # check file names
    if file_check>0:
        status = check_filenames(allfiles)
        if status!='ok':
            print(status, 'Files found:', allfiles, sep='\n')
            return
        elif file_check>1:
            print(status, 'Files found:', allfiles, sep='\n')
            if input('continue? [y/n]')=='n':
                return
    
    print('Reading files:')
    count = len(allfiles)
    i = 1
    
    with h5py.File(directory+'Tests.h5') as test_store:
        stored_tests = list(test_store.keys())
    
    for filename in allfiles:
        per = i/count*100
        print('\n') #clear the screen hack. if cmd, use os.system('cls')
        print('{:.0f} % Complete'.format(per))
        l = ['|']*i+[' ']*(count-i)+['| {:>2}/{}'.format(i, count)]
        print(''.join(l))
        
        new_name = re.search('\w{4}-\d{3,4}(_\d+)?',filename).group().replace('-','N')
#        if new_name in stored_tests:
#            print(new_name + ' already in keys! skipping...')
#            i = i + 1
#            continue
        if filename.endswith(('.xls','.xlsx')):
            testframe = read_excel(directory+filename)
        elif filename.endswith(('.csv')) and ('channel_names' not in filename) and ('test_names' not in filename):
            testframe = pandas.read_csv(directory+filename, dtype=numpy.float64)
        
        if data_check>0:
            status = check_testframe(testframe)
            if status!='ok':
                print(new_name,status,'df size:', testframe.shape,'df columns:',testframe.columns, sep='\n')
                print(testframe)
                return
                    
            if data_check>1:
                print(new_name,status,'df size:', testframe.shape,'df columns:',testframe.columns, sep='\n')
                print(testframe)
                if input('continue? [y/n]')=='n':
                    return
        testframe.columns = ['X' + i for i in testframe.columns]
        types = [(i, numpy.float64) for i in testframe.columns]
        
        
        with h5py.File(directory+'Tests.h5') as test_store:
            ds = test_store.create_dataset(new_name,shape=(testframe.shape[0],),dtype=types)
            ds[...] = testframe.apply(tuple,axis=1).values
            
        i = i+1

def write_angle(directory):
    allfiles = os.listdir(directory)
    print('Reading files:')
    count = len(allfiles)
    i = 1
    for filename in allfiles:
        if filename.endswith(('.xls','.xlsx')):
            book = xw.Book(directory+filename)
            testframe = pandas.DataFrame(book.sheets[0].range('A4').expand().value,
                                         columns=['Time','Up_x','Up_y','Down_x','Down_y'])
            book.close()
        else:
            continue
            
        new_name = re.search('\w{4}-\d{3,4}(_\d+)?',filename).group().replace('-','N')
        types = [(i, numpy.float64) for i in testframe.columns]
        with h5py.File(directory+'Tests.h5') as test_store:
            if new_name in list(test_store.keys()):
                print(new_name+' already in keys! skipping write...')
                continue
            ds = test_store.create_dataset(new_name,shape=(testframe.shape[0],),dtype=types)
            ds[...] = testframe.apply(tuple,axis=1).values
        
#%%%
if __name__=='__main__':
    from PMG.read_data import update_test_info
    
    def write_tc():
        directory_tc = 'P:\\Data Analysis\\Data\\TC\\'
        writeHDF5(directory_tc, file_check=1, data_check=1)
        update_test_info()
    
    def write_se():
        directory_se = 'P:\\Data Analysis\\Data\\SE\\'
        writeHDF5(directory_se)
        update_test_info()
    
    def write_angle_213():
        directory_angle = 'P:\\Data Analysis\\Data\\angle_213\\'
        write_angle(directory_angle)
    
    write_tc()
#    write_angle_213()
#    write_se()