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

def writeHDF5(directory, chlist):
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
    csvfiles = []
    xlsonly = []
    for filename in allfiles:
        if filename.endswith('.csv'):
            csvfiles.append(filename)
        converted = any([filename[:filename.find('.')] in file for file in csvfiles])
        if filename.endswith(('.xls','.xlsx')) and not converted:
            xlsonly.append(filename)

    if len(xlsonly) > 0:

        print('Converting to csv. This will take time but only once:')
        count = len(xlsonly)
        i = 1
        for filename in xlsonly:
            print(directory+'/'+filename)
            testframe = pandas.read_excel(directory+'/'+filename,
                                          sheetname=None, header=0,
                                          index_col=0, skiprows=[1,2])
            testframe = pandas.concat(list(testframe.values()), axis=1)
            try:
                start = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], -0.01001))
                end = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], 0.3999))+1
            except KeyError as e:
                raise KeyError('{} of test: {}. File has a different arangement '\
                               'of data or is missing time column'.format(e, filename))
            # keep measurements that have only one value (e.g. HIC)
            if start>1:
                single_value = testframe[testframe.columns[testframe.mask(testframe==0).count()==1]].iloc[0,:]
                testframe.update(single_value.to_frame(name=start+1).T)
                
            testframe = testframe.iloc[start:end,:]
            testframe.to_csv(directory+filename[:-4]+'.csv', index=False)
            per = i/count*100
            i = i+1
            print('{:.0f} % Complete'.format(per))

            csvfiles.append(filename[:-4]+'.csv')

    with pandas.HDFStore(directory+'Tests.h5') as test_store,\
         pandas.HDFStore(directory+'Channels.h5') as ch_store:

        print('Reading csv:')
        count = len(csvfiles)
        i = 1
        for filename in csvfiles:

            header = pandas.read_csv(directory+'/'+filename, nrows=0, dtype=numpy.float64)
            cols = header.columns.intersection(chlist).tolist()+['T_10000_0']
            testframe = pandas.read_csv(directory+'/'+filename, usecols=cols, dtype=numpy.float64)

            per = i/count*100
            print('\n'*30) #clear the screen hack. if cmd, use os.system('cls')
            print('{:.0f} % Complete'.format(per))
            l = ['|']*i+[' ']*(count-i)+['| {:>2}/{}'.format(i, count)]
            print(''.join(l))
            i = i+1

            new_name = re.search('\w{4}-\d{3,4}(_\d+)?',filename).group().replace('-','N')

            test_store.put(new_name, testframe, format='table')

        for ch in chlist:

            framelist = testframe[['T_10000_0']] # [[]] returns dataframe instead of series
            framelist.columns = ['Time']
            tcnlist = []

            for key in test_store.keys():

                tcn = key[1:].replace('N','-')
                testdata = test_store.select(key, columns=chlist) # removed square brackets from columns=chlist

                if ch in testdata.columns:
                    chdata = testdata[[ch]]
                    chdata.columns = [tcn]
                    framelist = pandas.concat([framelist, chdata], axis=1)
                else:
                    tcnlist.append(tcn)

            if len(tcnlist) > 0:
                print('The channel '+ch+' wasn\'t in these TCNs:\n')
                print(', '.join(tcnlist)+'\n')

            ch_store.put('C'+ch, framelist, format='table')

#%%%
#import tables as tb
#tb.file._open_files.close_all()    #Close all open stores
