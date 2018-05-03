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

dummy = 'Y2'
plotfigs = 1
savefigs = 1
writefiles = 1


if dummy=='Y2':
    channels = ['Head 3ms','Chest 3ms']
    exclude = ['SE16-1012_2',
               'SE16-1014_2',
               'SE17-1015_2',
               'SE17-1016_2',
               'SE16-0399',
               'SE16-0402',
               'SE16-0395',
               'SE16-0403',
               'SE17-1025_2',
               'SE17-1026_2']
else:
    channels = ['Head 3ms','Chest 3ms','Head Excursion','Knee Excursion','Seat Excursion','Seat-Head']
    exclude = []

table = tb.get('SLED')
table_y7 = table.query('DUMMY==\'' + dummy + '\'').filter(items=['SE','MODEL','INSTALL_2','SLED']+channels)
table_y7 = table_y7.set_index('SE',drop=True)
if not(exclude==[]):
    table_y7 = table_y7.drop(exclude,axis=0)
models = np.unique(table_y7['MODEL'])
sleds = np.unique(table_y7['SLED'])
types = table_y7['INSTALL_2'].dropna().unique()
writename = 'C:\\Users\\tangk\\Python\\Sled_' + dummy + '_'

#%% plot mean +/- std and distributions
if plotfigs:
    # compare across sleds for each model and installation type
    for m in models:
        for t in types:
            for i,ch in enumerate(channels):
                x = {}
                for s in sleds:
                    se = table_y7.query('MODEL==\''+m+'\' and SLED==\''+s+'\' and INSTALL_2==\'' + t + '\'').index
                    if len(se)==0:
                        x[s] = [np.nan]
                    else:
                        xs = table_y7[ch][list(se)]
                        if len(xs)>0:
                            x[s] = xs
                        else:
                            x[s] = [np.nan]
                if np.isnan(x['new_accel']).all() and np.isnan(x['new_decel']).all() and np.isnan(x['old_accel']).all():
                    continue
                fig = plt.figure(figsize=(5,5))
                ax = plt.axes()
                ax = plot_cat_nobar(ax,x)
                ax.set_title(ch + '\n ' + m + ' ' + t)
                if savefigs:
                    fig.savefig(writename + 'dot_' + ch + '_' + t + '_' + m + '.png', bbox_inches='tight')
                plt.show()
                plt.close(fig) 

#%% get ratios--method B
meanprops = {}

col1 = np.matlib.repmat(np.asarray(channels).reshape(-1,1),1,len(sleds)).flatten()
col2 = np.matlib.repmat(sleds,1,len(channels)).flatten()
mp = pd.DataFrame(index=models,columns=[col1,col2])
for i, ch in enumerate(channels):
    for t in types:
        for m in models:
            for s in sleds:
                se = table_y7.query('MODEL==\''+m+'\' and SLED==\''+s+'\' and INSTALL_2==\'' + t + '\'').index
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
            ax.set_title(ch + ' ' + t)
            if savefigs:
                fig.savefig(writename + 'bar_' + ch + '_' + t + '.png', bbox_inches='tight')
            plt.show()
            plt.close(fig)
        
        log_meanprops = mp[ch].applymap(np.log)
        if writefiles:
            log_meanprops.to_csv(writename + 'log_meanprops_' + ch + '_' + t + '.csv')
            mp[ch].to_csv(writename + 'meanprops_' + ch + '_' + t + '.csv')
        display(mp[ch])
        display(ch + ' ' + t + ':')
        display(np.nanmean(mp[ch]['old_accel'].astype(float)))
        display(np.nanstd(mp[ch]['old_accel'].astype(float)))