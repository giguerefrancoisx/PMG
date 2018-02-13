# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:56:11 2017

@author: giguerf
"""

import os
import matplotlib.pyplot as plt
import pandas
import math
from GitHub.COM import openbook as ob
from collections import OrderedDict
#from lookup_pop import lookup_pop
#from foreign_code import useful_function

plt.close('all')
readdir = os.fspath('P:/BOOSTER/SAI')
savedir = os.fspath('P:/BOOSTER/Plots/')
colors = {'Chest':'tab:blue','Pelvis':'tab:green','Chest2':'tab:red','Pelvis2':'tab:purple'}
sets = {}
fulldata = {}
places = {'CHST':'Chest','PELV':'Pelvis'}
groupnames = {'A':'Clek','B':'Bubblebum','C':'Evenflo Evolve','D':'Takata + Mifold','E':'All Others'}
groupdict = {}

#%% SECTION 1 - Open channel workbooks and group pairs in main loop

#For each channel workbook in the read directory, a dataframe is made. The data is sorted
#by population group: A - Clek, B - Bubblebum, C - Evolve, D - Mifold+Takata, E - All others
#as well as by crash conditions: speed, method. Plotting dataframes are created for the groups
#and conditions. 

for filename in os.listdir(readdir):
    if (filename[2:6] == 'CHST') or (filename[2:6] == 'PELV'):
        
        chdata = pandas.read_excel(readdir+'/'+filename)
        fulldata[places[filename[2:6]]] = chdata
        
time = chdata.iloc[:,0] #rename time channel for legibility
tot = pandas.concat(list(fulldata.values()), axis = 1)
bounds = (0,0.2,math.floor(min(tot.min())),math.ceil(max(tot.max())))
ylabel = 'Acceleration (X-direction) [g]'
title = ['Type: Offset, Speed: 48 km/h','Type: Mur, Speed: 48 km/h','Type: Offset, Speed: 56 km/h','Type: Mur, Speed: 56 km/h']
    
for group in list(groupnames.keys()):
    
    groupdata = {'Chest':[],'Pelvis':[]}
    
    for place in list(places.values()):
        
        chdata = fulldata[place]
        [genpop, cutpop] = ob.lookup_pop(chdata.columns[1:].tolist(),group)
        sets[group] = cutpop
    
        cut = chdata[cutpop.ALL]
        offset = cutpop[cutpop['Type'] == 'Frontale/Véhicule']
        mur = cutpop[cutpop['Type'] == 'Frontale/Mur']
        
        off48 = chdata[offset[offset['Vitesse'] == 48].ALL]
        off56 = chdata[offset[offset['Vitesse'] == 56].ALL]
        mur48 = chdata[mur[mur['Vitesse'] == 48].ALL]
        mur56 = chdata[mur[mur['Vitesse'] == 56].ALL]

        groupdata[place].extend([off48,mur48,off56,mur56])
        
    groupdict[group] = groupdata

#%% FIGURE 1

    plt.figure(groupnames[group]+' Grid', figsize=(20, 12.5))
    plt.suptitle(groupnames[group]+' Relative Chest/Pelvis Accelerations')
    
    for place in list(places.values()):
        for j in range(4):
            
            plt.subplot(2,2,j+1)
            
            if groupdict[group][place][j].shape[1] != 0:
                if group == 'D':
                    
                    cutpop = sets[group]
                    Takata = cutpop[cutpop['Marque'] == 'TAKATA']
                    
                    try:
                        plt.plot(time, groupdict[group][place][j][Takata.ALL], '-', color = colors[place+'2'], label = place)
                        plt.legend(loc = 4)
                    except KeyError:
                        plt.plot(time, groupdict[group][place][j], '-', color = colors[place], label = place)
                    
                    try:
                        plt.plot(time, groupdict[group][place][j].drop(Takata.ALL, axis = 1), '-', color = colors[place], label = place)
                    except ValueError:
                        plt.plot(time, groupdict[group][place][j], '-', color = colors[place], label = place)
                else:
                    plt.plot(time, groupdict[group][place][j], '-', color = colors[place], label = place)
            
            plt.axis(bounds)
            plt.title(title[j])
            plt.ylabel(ylabel)
            plt.xlabel('Time [s]')
            plt.annotate('n = %d' % groupdict[group][place][j].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
            #next three lines create a legend with no repeating entries
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc = 4)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig(savedir+'Grid_'+groupnames[group]+'.png', dpi = 600)

plt.close('all')
#%%
#peaks = pandas.concat([pandas.DataFrame(fulldata['Chest'].min(),columns = ['Chest']),pandas.DataFrame(fulldata['Pelvis'].min(),columns = ['Pelvis'])],axis = 1)
#peaks.to_excel('P:/BOOSTER/Plots/Peaks.xlsx')