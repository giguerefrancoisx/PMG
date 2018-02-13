# -*- coding: utf-8 -*-
"""
PLOT FROM WORKBOOK
    Generates plots from channel workbooks

Created on Wed Oct 11 14:18:22 2017

@author: giguerf
"""
def plotbook():
    import os
    import matplotlib.pyplot as plt
    import pandas
    import math
#    from collections import OrderedDict
    from lookup_pairs import lookup_pairs
    #from foreign_code import useful_function
    
    plt.close('all')
    readdir = os.fspath('C:/Users/giguerf/.spyder-py3/SAI')
    savedir = os.fspath('P:/SLED/Plots/')
    colors = {'CHST':'tab:blue','PELV':'tab:green','CHST2':'tab:orange','PELV2':'tab:purple'}
#    fulldata = []
    missing = []
    descriptions = pandas.read_excel('P:/AHEC/Descriptions.xlsx', index_col = 0)
    description = descriptions.transpose().to_dict('list')
    
    #%% SECTION 1 - Open channel workbooks and group pairs in main loop

    #For each channel workbook in the read directory, a dataframe is made. The data is grouped
    #into pairs of 'OLD' and 'NEW'. 
    #plotting dataframes are created for the groups and statistics (mean, std) are added  
    
    for filename in os.listdir(readdir):
        if filename.endswith(".xlsx"):
            
            chdata = pandas.read_excel(readdir+'/'+filename)
            pairs = lookup_pairs(chdata.columns[1:].tolist()) #assign pairs via external function
            time = chdata.iloc[:,0] #renamed time channel for legibility

            if chdata.shape[1] != 1:
                for tcn in pairs.NEW.tolist()+pairs.OLD.tolist():
                    if tcn not in chdata.columns.tolist():
                        missing.append(tcn)
                        chdata[tcn] = [float('NaN') for i in range(chdata.shape[0])] ###drop duplicates?
                        
                old = chdata[pairs.OLD]
                new = chdata[pairs.NEW]
                
                oldstats = pandas.DataFrame([old.mean(axis = 1), old.mean(axis = 1)+2*old.std(axis = 1), old.mean(axis = 1)-2*old.std(axis = 1)])
                oldstats = oldstats.transpose()
                oldstats = oldstats.rolling(window=30,center=False).mean().shift(-15)
                oldstats.columns = ['Mean','High','Low']
                
                newstats = pandas.DataFrame([new.mean(axis = 1), new.mean(axis = 1)+2*new.std(axis = 1), new.mean(axis = 1)-2*new.std(axis = 1)])
                newstats = newstats.transpose()
                newstats = newstats.rolling(window=30,center=False).mean().shift(-15)
                newstats.columns = ['Mean','High','Low']
                
#    #%% FIGURE 1
#                #First figure with three subplots: two for data (each group), one for mean and std dev
#                plt.figure('Groups: '+filename[:16], figsize=(20, 12.5))
#                plt.suptitle('Booster Seats Sled Tests\n'+filename[:16]+' - '+description[filename[:16]][0]) # or str(pairs.GROUPE.iloc[0])
#                
                tot = pandas.concat([old, new], axis = 1)
                bounds = (0,0.4,math.floor(min(tot.min())),math.ceil(max(tot.max())))
                
                if filename[12:14] == 'AC':
                    ylabel = 'Acceleration ('+filename[14:15]+'-direction) [g]'
                elif filename[12:14] == 'FO':
                    ylabel = 'Force ('+filename[14:15]+'-direction) [N]'
#                    
#                #FIRST SUBPLOT - 'OLD' Group full data set
#                plt.subplot(2,2,1)
#                plt.plot(time, old, '.', color = 'tab:blue', markersize=0.5, label = 'Data')
#                plt.axis(bounds)
#                plt.title('Old Sled Data')
#                plt.ylabel(ylabel)
#                plt.xlabel('Time [s]')
#                plt.annotate('n = %d' % old.dropna(axis = 1).shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
#                #next three lines create a legend with no repeating entries
#                handles, labels = plt.gca().get_legend_handles_labels()
#                by_label = OrderedDict(zip(labels, handles))
#                plt.legend(by_label.values(), by_label.keys(), loc = 4)
#        
#                #SECOND SUBPLOT - 'NEW' Group full data set
#                plt.subplot(2,2,3)
#                plt.plot(time, new, '.', color = 'tab:green', markersize=0.5, label = 'Data')
#                plt.axis(bounds)
#                plt.title('New Sled Data')
#                plt.ylabel(ylabel)
#                plt.xlabel('Time [s]')
#                plt.annotate('n = %d' % new.dropna(axis = 1).shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
#        
#                handles, labels = plt.gca().get_legend_handles_labels()
#                by_label = OrderedDict(zip(labels, handles))
#                plt.legend(by_label.values(), by_label.keys(), loc = 4)
#                
#                #THIRD SUBPLOT - 'OLD' vs 'NEW' Groups (mean and intervals)
#                plt.subplot(1,2,2)
#                plt.plot(time, oldstats['Mean'], color = 'tab:blue', label = 'Mean (Old)')
#                plt.fill_between(time, oldstats['High'], oldstats['Low'], color = 'tab:blue', alpha = 0.25, label = 'Intervals (Old)')
#                plt.plot(time, newstats['Mean'], color = 'tab:green', label = 'Mean (New)')
#                plt.fill_between(time, newstats['High'], newstats['Low'], color = 'tab:green', alpha = 0.25, label = 'Intervals (New)')
#                plt.axis(bounds)
#                plt.title('Old & New Sled Data (Mean and Intervals)')
#                plt.ylabel(ylabel)
#                plt.xlabel('Time [s]')
#                plt.legend(loc = 4)
#                plt.annotate('n = %d' % old.dropna(axis = 1).shape[1], (0.01, 0.03), xycoords='axes fraction') # compute sample size
#                plt.annotate('n = %d' % new.dropna(axis = 1).shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
#        
#                figManager = plt.get_current_fig_manager() # maximize window for visibility
#                figManager.window.showMaximized()
#                plt.savefig(savedir+'/'+'Groups_'+filename[:16]+'.png', dpi = 200)
        #%% FIGURE 2
                #Secondary figure - all pairs plotted individually
                plt.figure('All Pairs: '+filename[:16], figsize=(20, 12.5))
                plt.suptitle('Booster Seats Sled Tests\n'+filename[:16]+' - '+description[filename[:16]][0]) #put channel description
                
                s = math.ceil(math.sqrt(pairs.shape[0])) #set subplot size from number of pairs
                for i in range(pairs.shape[0]):
                    
                    plt.subplot(s,s,i+1) 
                    plt.plot(time, old.iloc[:,i], color = colors[filename[2:6]], label = old.columns[i]+', Old')
                    plt.plot(time, new.iloc[:,i], color = colors[filename[2:6]+'2'], label = new.columns[i]+', New')
                    plt.axis(bounds)
                    plt.title('%s' % pairs.iloc[i,2])
                    plt.ylabel(ylabel)
                    plt.xlabel('Time [s]')
                    plt.legend(loc = 4)
                         
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                plt.subplots_adjust(top = 0.93, bottom = 0.05, left = 0.04, right = 0.98, hspace = 0.42, wspace = 0.23)
                plt.savefig(savedir+'/'+'All_Pairs_'+filename[:16]+'.png', dpi = 200)
            else:
                print('Channel Workbook was blank!? '+filename[:16])
    #plt.show()
    plt.close('all')
    
    for tcn in list(set(missing)):
        print('Oops! '+tcn+' was not plotted. It\'s missing from the data folder or its pair had different channels')
