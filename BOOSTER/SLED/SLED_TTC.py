# -*- coding: utf-8 -*-
"""
TEST TO CHANNELS
    Pulls individual channel lines from the directory of TCNs and produces workbook for each channel
    
Created on Wed Oct 11 08:35:04 2017

@author: giguerf
"""
def writebook(chlist):
    import os
    import pandas   
    #%% SECTION 1 - import TCN workbooks into dataframes
    
    #The main for-loop opens files in the directory that are .xls files and reads all sheets
    #into a pandas dataframe. If multiple sheets are read, the dataframe returns a dictionary
    #type and must be reconfigured into a single dataframe. Each new frame is appended to the
    #list of frames (dictionary). A counter is included to show progress as read_excel() is slow.
    
    #(optional) make sure each pair has both SAI files in the set
    
    testframedict = {}
    directory = os.fspath('P:/SLED/Data/')
    count = len(os.listdir(directory))
    i = 1
    for filename in os.listdir(directory):
        if filename.endswith('.xls'):
            testframe = pandas.read_excel(directory+'/'+filename, sheetname = None, header = 0,index_col = 0,skiprows = [1,2])
            
            if len(testframe) == 1:
                testframe = list(testframe.items())[0][1]
            else:
                testframe = pandas.concat([list(testframe.items())[0][1],list(testframe.items())[1][1]], axis = 1)
            
            testframedict[filename[:-9]] = testframe #trim 9 characters from end, which preserves '_2' in TCNs
            per = i/count*100
            i = i+1
            print('%.1f %% Complete' % per)
            continue
        else:
            continue

#    ### Delete previous data in save folder /.spyder-py3/SAI/
    for filename in os.listdir('C:/Users/giguerf/.spyder-py3/SAI/'):
        os.remove('C:/Users/giguerf/.spyder-py3/SAI/'+filename)
    #%% SECTION 2 -  For each channel in the list desired, extract that channel from the dataframes
                    #and compile the set into a channel workbook
                    
    #A channel list must be specified or included in the code. The main for-loop will iterate
    #over the channels, reading the list of dataframes for tests with that channel. Initially,
    #a dataframe is created from the 'time' column of the first test. Next, the channel lines 
    #read will be joined to this dataframe. If a channel is not found in a test, a warning prints.
    #Lastly, the column names are concatenated with the position number and output to excel
    #files in the directory specified
    
    data = {}
    for chname in chlist:
        framelist = testframe[['T_10000_0']] # for some reason [[]] returns dataframe instead of series!
        framelist.columns = ['Time']
        tcnlist = []

        for tcn in testframedict:
            if chname in list(set(chlist).intersection(testframedict[tcn].columns)):
                testdata = testframedict[tcn][[chname]]
                testdata.columns = [tcn]
                framelist = pandas.concat([framelist,testdata], axis = 1)
            else:
                tcnlist.append(tcn)
        
        if len(tcnlist) > 0:      
            print('Oops! The channel '+chname+' wasn\'t in these TCNs:')
            print('\n%s\n' % ', '.join(map(str, tcnlist)))
        data[chname] = framelist
#        data[chname].columns = [str(col) + '_' + chname[:2] for col in data[chname].columns]
        
#    data1 = pandas.concat([data['14CHST0000Y7ACXC'], data['16CHST0000Y7ACXC'].iloc[:,1:]], axis = 1)
#    data2 = pandas.concat([data['14PELV0000Y7ACXA'], data['16PELV0000Y7ACXA'].iloc[:,1:]], axis = 1)
    data['12CHST0000Y7ACXC'].to_excel('C:/Users/giguerf/.spyder-py3/SAI/1XCHST0000Y7ACXC.xlsx', index = False)
    data['12PELV0000Y7ACXA'].to_excel('C:/Users/giguerf/.spyder-py3/SAI/1XPELV0000Y7ACXA.xlsx', index = False)