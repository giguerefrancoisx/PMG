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

def writebook(chlist, directory, subdir='', safe=True):
    # SECTION 1 - import TCN workbooks into dataframes
    """
    USE WRITEHDF5() WHEN POSSIBLE\n


    Input:
        chlist: list of channels (ISO codes) to pull
        directory: absolute path to data folder
        subdirs: list of subfolders to open
        safe: True if you want to confirm before deleting SAI folder's contents

    Output:
        Files in the SAI folder

    The main for-loop opens files in the directory that are .xls files and
    reads all sheets into a pandas dataframe. If multiple sheets are read,
    the dataframe returns a dictionary type and must be reconfigured into a
    single dataframe. Each new frame is appended to the list of frames
    (a dictionary). A counter is included to show progress as read_csv() takes
    time. Once this is complete, clean out all old files in the target folder
    (SAI) before writing to avoid plotting errors. (Errors can occur after
    modifying chlist)

    (optional) make sure each pair has both SAI files in the set
    """

    allfiles = os.listdir(directory+subdir)
    csvfiles = []
    xlsonly = []
    for filename in allfiles:
        if filename.endswith('.csv'):
            csvfiles.append(filename)
        converted = any([filename[:-4] in file for file in csvfiles])
        if filename.endswith('.xls') and not converted:
            xlsonly.append(filename)

    if len(xlsonly) > 0:

        print('Converting to csv. This will take time but only once:')
        count = len(xlsonly)
        i = 1
        for filename in xlsonly:

            testframe = pandas.read_excel(directory+subdir+'/'+filename,
                                          sheetname=None, header=0,
                                          index_col=0, skiprows=[1,2])

            sheets = []
            for sheet in testframe.values():
                sheets.append(sheet)
            testframe = pandas.concat(sheets, axis=1)
#            testframe = testframe.dropna(axis=0, how='all')
            start = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], -0.01001))
            end = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], 0.3999))+1
            testframe = testframe.iloc[start:end,:]

            testframe.to_csv(directory+subdir+filename[:-4]+'.csv', index=False)
            per = i/count*100
            i = i+1
            print('{:.0f} % Complete'.format(per))

            csvfiles.append(filename[:-4]+'.csv')

    else:
        pass

    print('Reading csv:')
    testframedict = {}
    count = len(csvfiles)
    i = 1
    for filename in csvfiles:

        header = pandas.read_csv(directory+subdir+'/'+filename, nrows=0)
        cols = header.columns.intersection(chlist).tolist()+['T_10000_0']
        testframe = pandas.read_csv(directory+subdir+'/'+filename, usecols=cols)

        start = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], -0.01001))
        end = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], 0.3999))+1
        testframe = testframe.iloc[start:end,:]
        testframe = testframe.reset_index(drop=True)

        testframedict[filename[:-9]] = testframe
        per = i/count*100
        print('\n'*30) #clear the screen hack. if cmd, use os.system('cls')
        print('{:.0f} % Complete'.format(per))
        l = ['|']*i+[' ']*(count-i)+['| {:>2}/{}'.format(i, count)]
        print(''.join(l))
        i = i+1

    newdir = directory.lower().replace('data','SAI').upper()
    if safe:
        confirm = input('This will delete everything in '+newdir+', ok?\n')
        if confirm == 'ok':
            print('Working...')
            for filename in os.listdir(newdir):
                os.remove(newdir+filename)
    else:
        print('Clearing '+newdir)
        for filename in os.listdir(newdir):
                os.remove(newdir+filename)

    # SECTION 2 - For each channel in the list desired, extract that channel
                  #from the dataframes and compile the set into a channel
                  #workbook
    """
    A channel list must be specified or included in the code. The main for-loop
    will iterate over the channels, reading the list of dataframes for tests
    with that channel. Initially, a dataframe is created from the 'time' column
    of the first test. Next, the channel lines read will be joined to this
    dataframe. If a channel is not found in a test, a warning prints.
    """

    data = {}
    for chname in chlist:

        framelist = testframe[['T_10000_0']] # for some reason [[]] returns dataframe instead of series!
        framelist.columns = ['Time']
        tcnlist = []

        for tcn in testframedict:

            if chname in list(set(chlist).intersection(testframedict[tcn].columns)):
                testdata = testframedict[tcn][[chname]]
                testdata.columns = [tcn]
                framelist = pandas.concat([framelist, testdata], axis = 1)
            else:
                tcnlist.append(tcn+'_'+chname[0:2])

        if len(tcnlist) > 0:
            print('The channel '+chname+' wasn\'t in these TCNs:')
            print('\n%s\n' % ', '.join(map(str, tcnlist)))

        data[chname] = framelist

    return data

def writeHDF5(directory, chlist):
    """
    Input:
    ----------
    chlist : list
        list of channels (ISO codes) to pull
    directory : string
        absolute path to data folder

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
    here they can be access by openHDF5() in PMG.COM.openbook
    """

    allfiles = os.listdir(directory)
    csvfiles = []
    xlsonly = []
    for filename in allfiles:
        if filename.endswith('.csv'):
            csvfiles.append(filename)
        converted = any([filename[:-4] in file for file in csvfiles])
        if filename.endswith('.xls') and not converted:
            xlsonly.append(filename)

    if len(xlsonly) > 0:

        print('Converting to csv. This will take time but only once:')
        count = len(xlsonly)
        i = 1
        for filename in xlsonly:

            testframe = pandas.read_excel(directory+'/'+filename,
                                          sheetname=None, header=0,
                                          index_col=0, skiprows=[1,2])

            sheets = []
            for sheet in testframe.values():
                sheets.append(sheet)
            testframe = pandas.concat(sheets, axis=1)
            start = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], -0.01001))
            end = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], 0.3999))+1
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

            start = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], -0.01001))
            end = int(numpy.searchsorted(testframe.loc[:,'T_10000_0'], 0.3999))+1
            testframe = testframe.iloc[start:end,:]
            testframe = testframe.reset_index(drop=True)

            per = i/count*100
            print('\n'*30) #clear the screen hack. if cmd, use os.system('cls')
            print('{:.0f} % Complete'.format(per))
            l = ['|']*i+[' ']*(count-i)+['| {:>2}/{}'.format(i, count)]
            print(''.join(l))
            i = i+1

            test_store.put(filename[:-9].replace('-','_'), testframe, format='table')

        for chname in chlist: #this loop got slower

            framelist = testframe[['T_10000_0']] # for some reason [[]] returns dataframe instead of series!
            framelist.columns = ['Time']
            tcnlist = []

            for key in test_store.keys():

                tcn = key[1:].replace('_','-')
                testdata = test_store.select(key, columns=[chlist])
                if chname in list(set(chlist).intersection(testdata.columns)):
                    testdata = testdata[[chname]]
                    testdata.columns = [tcn]
                    framelist = pandas.concat([framelist, testdata], axis = 1)
                else:
                    tcnlist.append(tcn+'_'+chname[0:2])

            if len(tcnlist) > 0:
                print('The channel '+chname+' wasn\'t in these TCNs:')
                print('\n%s\n' % ', '.join(map(str, tcnlist)))

            ch_store.put('C'+chname, framelist.iloc[:4100], format='table')
#%%%
#import tables as tb
#tb.file._open_files.close_all()    #Close all open stores
