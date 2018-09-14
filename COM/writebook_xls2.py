# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:04:56 2018

@author: tangk
"""

# -*- coding: utf-8 -*-
"""
TEST TO CHANNELS
    Pulls individual channel lines from the directory of TCNs and produces a
    workbook for each channel

Created on Wed Nov  1 10:51:26 2017

@author: giguerf
"""
import os
import pandas
import numpy
import re

def writeHDF5(directory):
    """
    Input:
    ----------
    directory : string
        absolute path to data folder (include trailing slash)
    chlist : list
        list of channels (ISO codes) to pull


    Output:
    ----------
    None. Generates HDF5 stores in the chosen directory with the data

    Notes on function:
    ----------
    The main for-loop opens files in the directory that are .xls files and
    converts them to csv files for faster access. (This may not be necessary
    with HDF5 stores which are semi-permanent). First, the files are read one
    by one into a pandas dataframe. If multiple sheets are read, the dataframe
    returns a dictionary type and must be reconfigured into a single dataframe.
    Each new frame is stored in the 'Tests' HDF5 store. A counter is included
    to show progress as read_csv() takes time. Once this is complete, the files
    are rearranged by channel and stored in the 'Channels' HDF5 store. From
    here they can be accessed by openHDF5() in PMG.COM.openbook or better yet
    by import_data() in PMG.COM.data
    """

    allfiles = os.listdir(directory)
    xlsonly = []
    csvfiles = []
    for filename in allfiles:
        if filename.endswith(('.xls','.xlsx')):# and not converted:
            xlsonly.append(filename)
        elif filename.endswith('.csv'):
            if not ('channel_names' in filename or 'test_names' in filename):
                csvfiles.append(filename)


    with pandas.HDFStore(directory+'Tests.h5') as test_store:

        print('Reading files:')
        count = len(xlsonly)
        i = 1
        for filename in xlsonly:
            testframe = pandas.read_excel(directory+filename,
                                          sheet_name=None, header=0,
                                          index_col=0, skiprows=[1,2])
            testframe = pandas.concat(list(testframe.values()), axis=1)
            try:
                start = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], -0.01001))
                end = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], 0.3999))+1
            except KeyError as e:
                raise KeyError('{} of test: {}. File has a different arangement '\
                               'of data or is missing time column'.format(e, filename))

            testframe = testframe.iloc[start:end,:]

            per = i/count*100
            print('\n'*30) #clear the screen hack. if cmd, use os.system('cls')
            print('{:.0f} % Complete'.format(per))
            l = ['|']*i+[' ']*(count-i)+['| {:>2}/{}'.format(i, count)]
            print(''.join(l))
            i = i+1

            new_name = re.search('\w{4}-\d{3,4}(_\d+)?',filename).group().replace('-','N')

            test_store.put(new_name, testframe, format='table')
            
        print('Reading csv:')
        count = len(csvfiles)
        i = 1
        for filename in csvfiles:
            testframe = pandas.read_csv(directory+'/'+filename, dtype=numpy.float64)

            per = i/count*100
            print('\n'*30) #clear the screen hack. if cmd, use os.system('cls')
            print('{:.0f} % Complete'.format(per))
            l = ['|']*i+[' ']*(count-i)+['| {:>2}/{}'.format(i, count)]
            print(''.join(l))
            i = i+1

            new_name = re.search('\w{4}-\d{3,4}(_\d+)?',filename).group().replace('-','N')

            test_store.put(new_name, testframe, format='table')
#%%%
#import tables as tb
#tb.file._open_files.close_all()    #Close all open stores
if __name__=='__main__':
    from PMG.read_data import update_test_info
    directory = 'P:\\Data Analysis\\Data\\'
    writeHDF5(directory)
    update_test_info()
