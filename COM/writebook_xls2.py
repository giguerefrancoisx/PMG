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

def writeHDF5(directory):

    allfiles = os.listdir(directory)
#    xlsonly = []
#    csvfiles = []
#    for filename in allfiles:
#        if filename.endswith(('.xls','.xlsx')):# and not converted:
#            xlsonly.append(filename)
#        elif filename.endswith('.csv'):
#            if not ('channel_names' in filename or 'test_names' in filename):
#                csvfiles.append(filename)
    print('Reading files:')
    count = len(allfiles)
    i = 1
    for filename in allfiles:
        if filename.endswith(('.xls','.xlsx')):
            testframe = pandas.read_excel(directory+filename,sheet_name=None, header=0,index_col=0, skiprows=[1,2],dtype=numpy.float64)
            testframe = pandas.concat(list(testframe.values()), axis=1)
            try:
                start = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], -0.01001))
                end = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], 0.3999))+1
            except KeyError as e:
                raise KeyError('{} of test: {}. File has a different arangement '\
                               'of data or is missing time column'.format(e, filename))
            testframe = testframe.iloc[start:end,:]
        elif filename.endswith(('.csv')) and (not 'channel_names' in filename) and (not 'test_names' in filename):
            testframe = pandas.read_csv(directory+'/'+filename, dtype=numpy.float64)
        else:
            continue

        per = i/count*100
        print('\n'*30) #clear the screen hack. if cmd, use os.system('cls')
        print('{:.0f} % Complete'.format(per))
        l = ['|']*i+[' ']*(count-i)+['| {:>2}/{}'.format(i, count)]
        print(''.join(l))
        i = i+1
        
        new_name = re.search('\w{4}-\d{3,4}(_\d+)?',filename).group().replace('-','N')
        testframe.columns = ['X' + i for i in testframe.columns]
        types = [(i, numpy.float64) for i in testframe.columns]
        
        with h5py.File(directory+'Tests.h5') as test_store:
            if new_name in list(test_store.keys()):
                print(new_name + ' already in keys! skipping write...')
                continue
            ds = test_store.create_dataset(new_name,shape=(testframe.shape[0],),dtype=types)
            ds[...] = testframe.apply(tuple,axis=1).values

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
        writeHDF5(directory_tc)
        update_test_info()
    
    def write_se():
        directory_se = 'P:\\Data Analysis\\Data\\SE\\'
        writeHDF5(directory_se)
        update_test_info()
    
    def write_angle_213():
        directory_angle = 'P:\\Data Analysis\\Data\\angle_213\\'
        write_angle(directory_angle)
    
    write_angle_213()