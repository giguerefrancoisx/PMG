# -*- coding: utf-8 -*-
"""
PLOT FROM WORKBOOK
    Generates plots from channel workbooks

Created on Wed Oct 11 14:18:22 2017

@author: giguerf
"""
#def plotbook(subdir):
if 1==1:
    import os
    import matplotlib.pyplot as plt
    import math
    from GitHub.COM import openbook as ob
    from GitHub.COM import plotstyle as style
    from collections import OrderedDict
    #from foreign_code import useful_function
    
    plt.close('all')
    readdir = os.fspath('P:/BOOSTER/SAI')
    savedir = os.fspath('P:/BOOSTER/Plots/')
    subdir = os.fspath('H3/')
    
    #%% SECTION 1 - Open channel workbooks create plotting dataframes
    
    keys = [filename[:16] for filename in os.listdir(readdir)]
    values = ['Chest', 'Illiac_LowerL', 'Illiac_UpperL', 'Illiac_LowerR', 'Illiac_UpperR', 'Lumbar_X', 'Lumbar_Z', 'Pelvis']
    places = dict(zip(keys, values))
    colors = style.colordict(keys)
#    time = [i/10000 for i in range(-100,4000)]

    time, gendict, cutdict, genpop, cutpop = ob.gencut(readdir, 'D')

    #%% FIGURE 1 - Plot a comparison of the general population and the subset population
    
    plt.figure('General vs Subset', figsize=(20, 12.5))
    plt.suptitle('Booster Seat in Crash Test: Groups')
    
    bounds = style.bounds(measure = 'AC')
    ylabel = style.ylabel('AC','X')
    L = [1,2]
    M = [4,5]
    N = [3,6]
    i = 0
    
    for place in ['14CHST0000Y7ACXC','14PELV0000Y7ACXA']:
    #FIRST & SECOND subplots: General pop. data, Chest & Pelvis
        plt.subplot(2,3,L[i])
        plt.plot(time, gendict[place], '-', color = colors[place], markersize=0.5, label = 'Data')
        title = 'General Population Data, '+places[place]
        style.labels(title, bounds, ylabel)
        style.legend(4)
        plt.annotate('n = %d' % gendict[place].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
        

    #THIRD & FOURTH subplots: Subset pop. data, Chest & Pelvis
        plt.subplot(2,3,M[i])
        plt.plot(time, cutdict[place], '-', color = colors[place], markersize=0.5, label = 'Data')
        title = 'Subset Population Data, '+places[place]
        style.labels(title, bounds, ylabel)
        style.legend(4)
        plt.annotate('n = %d' % cutdict[place].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
        i = i + 1
    
    #FIFTH SUBPLOT - General pop. Chest vs. Pelvis means only
    plt.subplot(2,3,3)
    plt.plot(time, gendict['14CHST0000Y7ACXC_stats']['Mean'], color = 'tab:blue', label = 'Mean (Chest)')
    plt.plot(time, gendict['14PELV0000Y7ACXA_stats']['Mean'], color = 'tab:green', label = 'Mean (Pelvis)')
    title = 'General Population Chest vs. Pelvis data'
    style.labels(title, bounds, ylabel)
    style.legend(4)
    plt.annotate('n = %d' % gendict['14CHST0000Y7ACXC'].shape[1], (0.01, 0.05), xycoords='axes fraction') # compute sample size
    plt.annotate('n = %d' % gendict['14PELV0000Y7ACXA'].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
    
    #SIXTH SUBPLOT - Subset pop. Chest vs. Pelvis means only
    plt.subplot(2,3,6)
    plt.plot(time, cutdict['14CHST0000Y7ACXC_stats']['Mean'], color = 'tab:blue', label = 'Mean (Chest)')
    plt.plot(time, cutdict['14PELV0000Y7ACXA_stats']['Mean'], color = 'tab:green', label = 'Mean (Pelvis)')
    title = 'Subset Chest vs. Pelvis data'
    style.labels(title, bounds, ylabel)
    style.legend(4)
    plt.annotate('n = %d' % cutdict['14CHST0000Y7ACXC'].shape[1], (0.01, 0.05), xycoords='axes fraction') # compute sample size
    plt.annotate('n = %d' % cutdict['14PELV0000Y7ACXA'].shape[1], (0.01, 0.01), xycoords='axes fraction')
    
    style.save(savedir+subdir, 'Group_stats.png')
    
    #%% FIGURE 2 - Plot a comparison of responses by brand
        
    plt.figure('Chest vs Pelvis by Brand', figsize=(20, 12.5))
    plt.suptitle('Chest vs Pelvis by Brand')
    
    marques = genpop['Marque'].drop_duplicates().tolist()
    s = math.ceil(math.sqrt(len(marques)))
    
    for place in ['14CHST0000Y7ACXC','14PELV0000Y7ACXA']:
        i = 0 # index for 'marque'
        
        for marque in marques:
            TCNs = genpop[genpop['Marque'] == marque].ALL.tolist()
            TCNs = list(set(TCNs))
            genplot = gendict[place][TCNs]#.mean(axis = 1)
            
            plt.subplot(s,s,i+1)
            plt.plot(time, genplot, color = colors[place], label = place)
            plt.axis(bounds)
            plt.title(marque)
            plt.ylabel(ylabel)
            plt.xlabel('Time [s]')
            plt.annotate('n = %d' % genplot.shape[1], (0.01, 0.01), xycoords='axes fraction')
                
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc = 4)
            i = i+1
                
        TCNs = cutpop[cutpop['Marque'] == 'MIFOLD'].ALL.tolist()
        TCNs = list(set(TCNs))
        cutplot = cutdict[place][TCNs]#.mean(axis = 1)

        plt.subplot(s,s,i+1)
        plt.plot(time, cutplot, color = colors[place], label = place)
        plt.axis(bounds)
        plt.title('MIFOLD')
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.annotate('n = %d' % cutplot.shape[1], (0.01, 0.01), xycoords='axes fraction')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 4)
        
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.subplots_adjust(top = 0.93, bottom = 0.05, left = 0.06, right = 0.96, hspace = 0.46, wspace = 0.27)
        
    plt.savefig(savedir+subdir+'Brands_stats.png', dpi = 200)
    
    #%% FIGURE 3 - 
#    plt.figure('General vs Subset 2', figsize=(20, 12.5))
#    plt.suptitle('Booster Seat in Crash Test: Groups')
    
    ylabel = 'Force (X-direction) [N]'
    L = [1,4,7,10]
    M = [2,5,8,11]
    N = [3,6,9,12]
    i = 0
    
#    plt.subplots(4,3,sharey = 'all', figsize=(20, 12.5))
    for place in ['14ILACLELOY7FOXB','14ILACLEUPY7FOXB','14ILACRILOY7FOXB','14ILACRIUPY7FOXB']:
    #FIRST & SECOND subplots: General pop. data, Chest & Pelvis
        plt.subplot(4,3,L[i])
        plt.plot(time, gendict[place], '-', color = colors[place], markersize=0.5, label = 'Data')
        plt.xlim([0, 0.2])
        plt.title('General Population Data, '+places[place])
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.annotate('n = %d' % gendict[place].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
        #next three lines create a legend with no repeating entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 4)

    #THIRD & FOURTH subplots: Subset pop. data, Chest & Pelvis
        plt.subplot(4,3,M[i])
        plt.plot(time, cutdict[place], '-', color = colors[place], markersize=0.5, label = 'Data')
        plt.xlim([0, 0.2])
        plt.title('Subset Population Data, '+places[place])
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.annotate('n = %d' % cutdict[place].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
        #next three lines create a legend with no repeating entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 4)
        
    
        #FIFTH SUBPLOT - General pop. Chest vs. Pelvis means only
        plt.subplot(4,3,N[i])
        plt.plot(time, gendict[place+'_stats']['Mean'], color = 'tab:blue', label = 'Mean (General)')
    #    plt.fill_between(time, gendict['Chest_stats']['High'], gendict['Chest_stats']['Low'], color = 'tab:blue', alpha = 0.25, label = 'Intervals (General)')
        plt.plot(time, cutdict[place+'_stats']['Mean'], color = 'tab:green', label = 'Mean (Subset)')
    #    plt.fill_between(time, gendict['Pelvis_stats']['High'], gendict['Pelvis_stats']['Low'], color = 'tab:green', alpha = 0.25, label = 'Intervals (MIFOLD)')
        plt.xlim([0, 0.2])
        plt.title('General vs Subset '+places[place])
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.legend(loc = 4)
        plt.annotate('n = %d' % gendict[place].shape[1], (0.01, 0.05), xycoords='axes fraction') # compute sample size
        plt.annotate('n = %d' % cutdict[place].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
        i = i + 1
    
    minlim = 0
    maxlim = 0
    for i in range(12):
        plt.subplot(4,3,i+1)
        if plt.ylim()[0] < minlim: minlim = plt.ylim()[0]
        if plt.ylim()[1] > maxlim: maxlim = plt.ylim()[1]
    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.ylim([minlim, maxlim])
    
    figManager = plt.get_current_fig_manager() # maximize window for visibility
    figManager.window.showMaximized()
    plt.subplots_adjust(top = 0.93, bottom = 0.05, left = 0.06, right = 0.96, hspace = 0.23, wspace = 0.20)
    plt.savefig(savedir+subdir+'Group_stats_2.png', dpi = 200)
    
    #%% FIGURE 4
    plt.figure('General vs Subset 3', figsize=(20, 12.5))
    plt.suptitle('Booster Seat in Crash Test: Groups')
    
    ylabel = 'Force (X-direction) [N]'
    L = [1,2]
    M = [4,5]
    N = [3,6]
    i = 0
    
    for place in ['14LUSP0000Y7FOXA','14LUSP0000Y7FOZA']:
    #FIRST & SECOND subplots: General pop. data, Chest & Pelvis
        plt.subplot(2,3,L[i])
        plt.plot(time, gendict[place], '-', color = colors[place], markersize=0.5, label = 'Data')
        plt.xlim([0, 0.2])
        plt.title('General Population Data, '+places[place])
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.annotate('n = %d' % gendict[place].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
        #next three lines create a legend with no repeating entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 4)

    #THIRD & FOURTH subplots: Subset pop. data, Chest & Pelvis
        plt.subplot(2,3,M[i])
        plt.plot(time, cutdict[place], '-', color = colors[place], markersize=0.5, label = 'Data')
        plt.xlim([0, 0.2])
        plt.title('Subset Population Data, '+places[place])
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.annotate('n = %d' % cutdict[place].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
        #next three lines create a legend with no repeating entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 4)
    
        #FIFTH SUBPLOT - General pop. Chest vs. Pelvis means only
        plt.subplot(2,3,N[i])
        plt.plot(time, gendict[place+'_stats']['Mean'], color = 'tab:blue', label = 'Mean (General)')
    #    plt.fill_between(time, gendict['Chest_stats']['High'], gendict['Chest_stats']['Low'], color = 'tab:blue', alpha = 0.25, label = 'Intervals (General)')
        plt.plot(time, cutdict[place+'_stats']['Mean'], color = 'tab:green', label = 'Mean (Subset)')
    #    plt.fill_between(time, gendict['Pelvis_stats']['High'], gendict['Pelvis_stats']['Low'], color = 'tab:green', alpha = 0.25, label = 'Intervals (MIFOLD)')
        plt.xlim([0, 0.2])
        plt.title('General vs Subset '+places[place])
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.legend(loc = 4)
        plt.annotate('n = %d' % gendict[place].shape[1], (0.01, 0.05), xycoords='axes fraction') # compute sample size
        plt.annotate('n = %d' % cutdict[place].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
        i = i + 1
    
    minlim = 0
    maxlim = 0
    for i in range(6):
        plt.subplot(2,3,i+1)
        if plt.ylim()[0] < minlim: minlim = plt.ylim()[0]
        if plt.ylim()[1] > maxlim: maxlim = plt.ylim()[1]
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.ylim([minlim, maxlim])
    
    figManager = plt.get_current_fig_manager() # maximize window for visibility
    figManager.window.showMaximized()
    plt.subplots_adjust(top = 0.93, bottom = 0.05, left = 0.06, right = 0.96, hspace = 0.23, wspace = 0.20)
    plt.savefig(savedir+subdir+'Group_stats_3.png', dpi = 200)
    
#%% individual plots for new tests
    
    #ADD dict with subplot location
        #pair chest, pelv and ilac L/R
    subloc = {}
    subloc = dict(zip(list(places.keys()), [1,4,5,4,5,2,3,1]))
    for TCN in ['TC16-127','TC12-004','TC17-205']:
        plt.figure(TCN, figsize=(20, 12.5))
        speed = str(genpop[genpop['ALL']==TCN+'_14'].Vitesse.tolist()[0])+' km/h'
        plt.suptitle('Position Comparison for Newest Tests:\n'+TCN+' at '+speed)
        for i, place in enumerate(list(places.keys())):
            plt.subplot(2,3,subloc[place])
            plt.title(places[place])
            plt.xlabel('Time [s]')
            plt.xlim([0, 0.2])
            try:
                label = places[place]+': Pos. 14: '+genpop[genpop['ALL']==TCN+'_14'].Marque.tolist()[0]
                plt.plot(time, gendict[place][TCN+'_14'], label = label)
            except:
                pass
            try:
                label = places[place]+': Pos. 16: '+cutpop[cutpop['ALL']==TCN+'_16'].Marque.tolist()[0]
                plt.plot(time, cutdict[place][TCN+'_16'], label = label)
            except:
                pass
            plt.legend(loc = 5)
            plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
        
        style.save(savedir+subdir, TCN+'.png')
        plt.close('all')
