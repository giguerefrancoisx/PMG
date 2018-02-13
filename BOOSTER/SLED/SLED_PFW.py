# -*- coding: utf-8 -*-
"""
PLOT FROM WORKBOOK
    Generates plots from channel workbooks

Created on Wed Oct 11 14:18:22 2017

@author: giguerf
"""
def plotbook():
#if 1==1:
    import os
    import matplotlib.pyplot as plt
    import pandas
    import math
    from collections import OrderedDict
    from lookup_pairs import lookup_pairs
    #from foreign_code import useful_function
    
    plt.close('all')
    readdir = os.fspath('P:/SLED/SAI/')
    savedir = os.fspath('P:/SLED/Plots/')
    colors = {'Chest':'tab:blue','Pelvis':'tab:green','Chest2':'tab:orange','Pelvis2':'tab:purple'}
    olddict = {}
    newdict = {}
    places = {'CHST':'Chest','PELV':'Pelvis'}
    missing = []

    #%% SECTION 1 - Open channel workbooks create plotting dataframes
    
    #For each channel workbook in the read directory, a dataframe is made. The data is grouped
    #into pairs of 'old' and 'new' sled tests via lookup table. Plotting dataframes are created
    #for the pairs and means are added
    
    for filename in os.listdir(readdir):
        if filename.endswith(".xlsx"):
            
            chdata = pandas.read_excel(readdir+'/'+filename)
            pairs = lookup_pairs(chdata.columns[1:].tolist())
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
                
                olddict[places[filename[2:6]]] = old
                olddict[places[filename[2:6]]+'_stats'] = oldstats
                
                newdict[places[filename[2:6]]] = new
                newdict[places[filename[2:6]]+'_stats'] = newstats
        
            else:
                continue

    #%% FIGURE 1 - Plot a comparison of the general population and the subset population
    
    plt.figure('Old vs New Sled', figsize=(20, 12.5))
    plt.suptitle('Booster Seat in Sled Test: Groups')
    
    tot = pandas.concat([old,new], axis = 1)
    bounds = (0,0.2,math.floor(min(tot.min())),math.ceil(max(tot.max())))
    ylabel = 'Acceleration (X-direction) [g]'
    i = 1
    
    for place in list(places.values()):
    #FIRST & SECOND subplots: General pop. data, Chest & Pelvis
        plt.subplot(2,2,i)
        plt.plot(time, olddict[place], '-', color = colors[place], markersize=0.5, label = 'Data')
        plt.axis(bounds)
        plt.title('Old Sled Data, '+place)
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.annotate('n = %d' % olddict[place].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
        #next three lines create a legend with no repeating entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 4)

    #THIRD & FOURTH subplots: Subset pop. data, Chest & Pelvis
        plt.subplot(2,2,i+2)
        plt.plot(time, newdict[place], '-', color = colors[place+'2'], markersize=0.5, label = 'Data')
        plt.axis(bounds)
        plt.title('New Sled Data, '+place)
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.annotate('n = %d' % newdict[place].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
        #next three lines create a legend with no repeating entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 4)
        i = i + 1
    
    figManager = plt.get_current_fig_manager() # maximize window for visibility
    figManager.window.showMaximized()
    plt.subplots_adjust(top = 0.93, bottom = 0.05, left = 0.06, right = 0.96, hspace = 0.23, wspace = 0.20)
    plt.savefig(savedir+'Group_stats.png', dpi = 200)

    #%% FIGURE 2
    plt.figure('Old vs New Sled Combined', figsize=(20, 12.5))
    plt.suptitle('Booster Seat in Sled Test: Combined')
    
    plt.plot(time, olddict['Chest_stats']['Mean'], color = 'tab:blue', label = 'Mean (Chest, Old)')
    plt.plot(time, newdict['Chest_stats']['Mean'], color = 'tab:orange', label = 'Mean (Chest, New)')
    plt.plot(time, olddict['Pelvis_stats']['Mean'], color = 'tab:green', label = 'Mean (Pelvis, Old)')
    plt.plot(time, newdict['Pelvis_stats']['Mean'], color = 'tab:purple', label = 'Mean (Pelvis, New)')
    plt.axis(bounds)
    plt.title('Old vs New Sled - Chest vs. Pelvis data')
    plt.ylabel(ylabel)
    plt.xlabel('Time [s]')
    plt.legend(loc = 4)
    plt.annotate('n = %d' % olddict['Chest'].shape[1], (0.01, 0.010), xycoords='axes fraction') # compute sample size
    plt.annotate('n = %d' % newdict['Chest'].shape[1], (0.01, 0.08), xycoords='axes fraction') # compute sample size
    plt.annotate('n = %d' % olddict['Pelvis'].shape[1], (0.01, 0.06), xycoords='axes fraction') # compute sample size
    plt.annotate('n = %d' % newdict['Pelvis'].shape[1], (0.01, 0.04), xycoords='axes fraction')
    
    figManager = plt.get_current_fig_manager() # maximize window for visibility
    figManager.window.showMaximized()
    plt.subplots_adjust(top = 0.93, bottom = 0.05, left = 0.06, right = 0.96, hspace = 0.23, wspace = 0.20)
    plt.savefig(savedir+'Combined_stats.png', dpi = 200)
    
    #%% FIGURE 3 - Plot a comparison of responses by brand
    for mode in ['Old', 'New']:   
        plt.figure('Chest vs Pelvis by Brand:'+mode+' Sled', figsize=(20, 12.5))
        plt.suptitle('Chest vs Pelvis by Brand')
        
        groups = pairs['GROUP'].drop_duplicates().tolist()
        groups.sort()
        s = math.ceil(math.sqrt(len(groups)))
    
    
        for place in list(places.values()):
            i = 0 # index for 'marque'
            
            for seatgroup in groups:
                plt.subplot(s,s,i+1)
                if mode == 'Old':
                    TCNs = pairs[pairs['GROUP'] == seatgroup].OLD.tolist()
                    TCNs = list(set(TCNs))
                    oldplot = olddict[place][TCNs]#.mean(axis = 1)
                    plt.plot(time, oldplot, color = colors[place], label = place+', Old')
                    plt.annotate('n = %d' % oldplot.shape[1], (0.01, 0.01), xycoords='axes fraction')
                else:
                    TCNs = pairs[pairs['GROUP'] == seatgroup].NEW.tolist()
                    TCNs = list(set(TCNs))
                    newplot = newdict[place][TCNs]#.mean(axis = 1)
                    plt.plot(time, newplot, color = colors[place+'2'], label = place+', New')
                    plt.annotate('n = %d' % newplot.shape[1], (0.01, 0.01), xycoords='axes fraction')
                plt.axis(bounds)
                plt.title(seatgroup)
                plt.ylabel(ylabel)
                plt.xlabel('Time [s]')
                    
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc = 4)
                i = i+1
            
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.subplots_adjust(top = 0.93, bottom = 0.05, left = 0.06, right = 0.96, hspace = 0.46, wspace = 0.27)
        plt.savefig(savedir+'Brands_stats_'+mode+'.png', dpi = 200)

    #%% FIGURE 4 & 5 - Plot all pairs individually
    
    for place in list(places.values()):

        plt.figure('All Pairs: '+place, figsize=(20, 12.5))
        plt.suptitle('Booster Seats Sled Tests\n'+place)
        
        s = math.ceil(math.sqrt(pairs.shape[0])) #set subplot size from number of pairs
        for i in range(pairs.shape[0]):
            
            plt.subplot(s,s,i+1) 
            plt.plot(time, olddict[place].iloc[:,i], color = colors[place], label = olddict[place].columns[i]+', Old')
            plt.plot(time, newdict[place].iloc[:,i], color = colors[place+'2'], label = newdict[place].columns[i]+', New')
            plt.axis(bounds)
            plt.title('%s' % pairs.iloc[i,2])
            plt.ylabel(ylabel)
            plt.xlabel('Time [s]')
            plt.legend(loc = 4)
                 
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.subplots_adjust(top = 0.93, bottom = 0.05, left = 0.04, right = 0.98, hspace = 0.42, wspace = 0.23)
        plt.savefig(savedir+'/'+'All_Pairs_'+place+'.png', dpi = 200)
    
    plt.close('all')