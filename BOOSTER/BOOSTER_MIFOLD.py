# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:56:11 2017

@author: giguerf
"""

import os
import matplotlib.pyplot as plt
import pandas
#import math
#from collections import OrderedDict
from lookup_pop import lookup_pop
#from foreign_code import useful_function

plt.close('all')
fulldata = []
directory = os.fspath('C:/Users/giguerf/.spyder-py3/SAI')
savedir = os.fspath('P:/BOOSTER/Plots/')
#savedir = os.fspath('C:/Users/giguerf/.spyder-py3/ErrorTest/')

#    descriptions = pandas.read_excel('P:/AHEC/Descriptions.xlsx', index_col = 0)
#    description = descriptions.transpose().to_dict('list')

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
#%% SECTION 1 - Open channel workbooks and group pairs in main loop
i = 0
gen_ch = []
genstats_ch = []
cut_ch = []
cutstats_ch = []

for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
        chdata = pandas.read_excel(directory+'/'+filename)
        fulldata.append(chdata) #fulldata variable is for debugging
#        pairs = lookup_pairs(chdata.columns[1:]) #assign pairs via external function
#        TCNs = ['TC12-218_16','TC14-230_16','TC14-503_14','TC14-503_16','TC15-102_14','TC15-206_14','TC15-209_14','TC16-172_14']
        [genpop, cutpop] = lookup_pop(chdata.columns[1:].tolist())
        time = chdata.iloc[:,0] #rename time channel for legibility
        
        #Add tests not in dataset with null list (to simplify exceptions)
#        missing = []
#        for tcn in pairs.BELIER.tolist()+pairs.CIBLE.tolist():
#            if tcn not in chdata.columns.tolist():
#                missing.append(tcn)
#                chdata[tcn] = [float('NaN') for i in range(chdata.shape[0])]
        
        #BELOW: split tests into groups, then compute means (rolling mean) and stddevs
#        gen = chdata[genpop.ALL]
        cut = chdata[cutpop.ALL] # also where type = x and speed = y
#        cib = chdata[pairs.CIBLE]
#        bel = chdata[pairs.BELIER]
#        genstats = pandas.DataFrame([gen.mean(axis = 1), gen.mean(axis = 1)+2*gen.std(axis = 1), gen.mean(axis = 1)-2*gen.std(axis = 1)])
#        genstats = genstats.transpose()
#        genstats = genstats.rolling(window=100,center=False).mean().shift(-50) 
#        cutstats = pandas.DataFrame([cut.mean(axis = 1), cut.mean(axis = 1)+2*cut.std(axis = 1), cut.mean(axis = 1)-2*cut.std(axis = 1)])
#        cutstats = cutstats.transpose()
#        cutstats = cutstats.rolling(window=100,center=False).mean().shift(-50)
        
#        gen_ch.append(gen)
#        genstats_ch.append(genstats)
#        cut_ch.append(cut)
#        cutstats_ch.append(cutstats)
#        i = i+1
#%% FIGURE 1
#First figure with three subplots: two for data (each group), one for mean and std dev
plt.figure('Mifold Grid', figsize=(20, 12.5))
#plt.suptitle('Booster Seat in Crash Test\n'+filename[:16]+' - '+description[filename[:16]][0]) # or str(pairs.GROUPE.iloc[0])
plt.suptitle('Mifold Relative Chest/Pelvis Accelerations')

bounds = (0,0.4,-80,30)
ylabel = 'Acceleration (X-direction) [g]'

#FIRST SUBPLOT - Offset, 48
plt.subplot(2,2,1)
plt.plot(time, gen_ch[0], '.', color = 'tab:blue', markersize=0.5, label = 'Data')
plt.axis(bounds)
plt.title('Type: Offset, Speed: 48 km/h')
plt.ylabel(ylabel)
plt.xlabel('Time [s]')
#plt.annotate('n = %d' % gen_ch[0].shape[1], (0.01, 0.01), xycoords='axes fraction') # compute sample size
#next three lines create a legend with no repeating entries
#handles, labels = plt.gca().get_legend_handles_labels()
#by_label = OrderedDict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), loc = 4)