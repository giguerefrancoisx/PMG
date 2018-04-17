# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:44:50 2018

@author: tangk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PMG.COM import table as tb
from PMG.COM.plotfuns import *

dummy = 'Y6'
plotfigs = 1
savefigs = 0
writefiles = 0


if dummy=='Y2':
    channels = ['Head 3ms','Chest 3ms']
    exclude = ['SE16-0395',
               'SE16-0399',
               'SE16-0402',
               'SE16-1012_2',
               'SE17-1015_2',
               'SE17-1025_2']
else:
    channels = ['Head 3ms','Chest 3ms','Head Excursion','Knee Excursion']
    exclude = []

table = tb.get('SLED')
table_y7 = table.query('DUMMY==\'' + dummy + '\'').filter(items=['SE','MODEL','SLED','Head 3ms','Chest 3ms','Head Excursion','Knee Excursion'])
table_y7 = table_y7.set_index('SE',drop=True)
if not(exclude==[]):
    table_y7 = table_y7.drop(exclude,axis=0)
models = np.unique(table_y7['MODEL'])
sleds = np.unique(table_y7['SLED'])
writename = 'C:\\Users\\tangk\\Python\\Sled_' + dummy + '_'

#%% plot mean +/- std and distributions
if plotfigs:
    # compare across sleds for each model
    for m in models:
        for i,ch in enumerate(channels):
            fig = plt.figure(figsize=(5,5))
            ax = plt.axes()
            x = {}
            for s in sleds:
                se = table_y7.query('MODEL==\''+m+'\' and SLED==\''+s+'\'').index
                if len(se)==0:
                    x[s] = [np.nan]
                else:
                    xs = table_y7[ch][list(se)]
                    if len(xs)>0:
                        x[s] = xs
                    else:
                        x[s] = [np.nan]
            ax = plot_cat_nobar(ax,x)
            ax.set_title(ch + '\n ' + m)
            if savefigs:
                fig.savefig(writename + ch + '_' + m + '_dot.png', bbox_inches='tight')
            plt.show()
            plt.close(fig) 

#%% get ratios--method B
meanprops = {}

col1 = np.matlib.repmat(np.asarray(channels).reshape(-1,1),1,len(sleds)).flatten()
col2 = np.matlib.repmat(sleds,1,len(channels)).flatten()
mp = pd.DataFrame(index=models,columns=[col1,col2])
for i, ch in enumerate(channels):
    for m in models:
        for s in sleds:
            se = table_y7.query('MODEL==\''+m+'\' and SLED==\''+s+'\'').index
            if len(se)==0:
                x = np.nan
            else:
                x = table_y7[ch][list(se)]
            mp.set_value(m,(ch,s),np.mean(x))
    mp[ch] = mp[ch].divide(mp[ch]['new_accel'],axis='rows')
    if plotfigs:
        fig = plt.figure(figsize=(5,5))       
        ax = plt.axes()
        ax = plot_bar(ax,mp[ch])
        ax.set_title(ch)
        if savefigs:
            fig.savefig(writename + ch + 'mean_' + '_bar.png', bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    log_meanprops = mp[ch].applymap(np.log)
    if writefiles:
        log_meanprops.to_csv(writename + 'log_meanprops_' + ch + '.csv')
meanprops = mp