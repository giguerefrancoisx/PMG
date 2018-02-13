# -*- coding: utf-8 -*-
"""
LAB BELT SCORE PLOT
    Compare the lap belt score to key data to establish any correlations

Created on Fri Nov 10 11:43:05 2017

@author: giguerf
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
if 'C:/Users/giguerf/Documents' not in sys.path:
    sys.path.insert(0, 'C:/Users/giguerf/Documents')
from GitHub.COM import openbook as ob
from GitHub.COM import plotstyle as style
from GitHub.COM import get_peak as gp

readdir = os.fspath('P:/BOOSTER/SAI/Y7')
savedir = os.fspath('P:/BOOSTER/Plots/Y7')

time, fulldata, *_ = ob.openbook(readdir)

#%%
scoretable = pd.read_excel('P:/BOOSTER/IIHS scores/lapscoretable.xlsx', 
                               index_col = None, parse_cols = 9)

table = pd.read_excel('P:/BOOSTER/boostertable.xlsx', index_col = 0)

titlelist = ['Chest Peak', 'Pelvis Peak', 'Chest Start', 'Pelvis Start',
           'Lumbar X Peak', 'Lumbar Z Peak','Lumbar X Start', 'Lumbar Z Start',
           'Time Between Peak Values (Chest - Pelvis)', 
           'Difference in Peak Values (Chest - Pelvis)']

subplot_peak = dict(zip(['14CHST0000Y7ACXC','14PELV0000Y7ACXA',
                         '14LUSP0000Y7FOXA','14LUSP0000Y7FOZA'],[1,2,5,6]))

subplot_peaktime = dict(zip(['14CHST0000Y7ACXC','14PELV0000Y7ACXA', 
                             '14LUSP0000Y7FOXA','14LUSP0000Y7FOZA'],[3,4,7,8]))
#### FOR ORANGE
table = pd.read_excel('P:/BOOSTER/boostertable.xlsx', index_col = 0)
import xlrd
faro = pd.DataFrame(columns = ['LBS'])
for file in os.listdir('P:/BOOSTER/Faro/'):
    tcn = file[:-4]
    for sheetname in ['BELT MEASURE POSITION 14','BELT MEASURE POSITION 16']:
        try:
            raw = pd.read_excel('P:/BOOSTER/Faro/'+tcn+'.xls', sheetname=sheetname, index_col=4)
        except xlrd.XLRDError:
            try:
                raw = pd.read_excel('P:/BOOSTER/Faro/'+tcn+'.xls', sheetname=sheetname.title(), index_col=4)
            except xlrd.XLRDError:
                pass
            else:
                raw = raw['X']
                relative = (raw.iloc[0]-raw).loc[['Top Lap Belt Left','Top Lap Belt Right']]
                faro.loc[tcn+'_'+sheetname[-2:]] = relative.mean(axis=0)
        else:
            raw = raw['X']
            relative = (raw-raw.iloc[0]).loc[['Top Lap Belt Left','Top Lap Belt Right']]
            faro.loc[tcn+'_'+sheetname[-2:]] = relative.mean(axis=0)
temp = pd.merge(table, faro, how = 'outer', left_index = True, right_index = True)
tcnlist = list(table.index)
peakdata = {}
for channel in fulldata:
    peakdata[channel] = pd.DataFrame(columns = ['t0','tp','start','peak'])
    for tcn in fulldata[channel].columns.intersection(tcnlist):
        peakdata[channel].loc[tcn] = gp.get_peak(time, fulldata[channel][tcn])

peaktable = pd.concat([peakdata['14CHST0000Y7ACXC'], peakdata['14PELV0000Y7ACXA']], axis=1)
peaktable.columns = ['Chest Start Time', 'Chest Peak Time', 'Chest 5% Peak', 'Chest Peak','Pelvis Start Time', 'Pelvis Peak Time', 'Pelvis 5% Peak', 'Pelvis Peak']
t = pd.merge(temp, peaktable, how='outer', left_index=True, right_index=True)
t = t[t['Type_Mannequin'] == 'HIII Enfant'].drop('Location', axis=1)
t = t.reset_index()
t.to_excel('C:/Users/giguerf/table.xlsx', index=False)

#%%
plt.close('all')
for vitesse in [40,48,56]:
    tcnlist = list(table[table['Vitesse'] == vitesse].index)
    peakdata = {}
    lbs = {}
    for channel in fulldata:
        peakdata[channel] = pd.DataFrame(columns = ['t0','tp','start','peak'])
        for tcn in fulldata[channel].columns.intersection(tcnlist):
            peakdata[channel].loc[tcn] = gp.get_peak(time, fulldata[channel][tcn])
            mask = scoretable.Modele == table.loc[tcn].Modele
            lbs[tcn] = scoretable[mask].iloc[0,-4:]

    axs = style.subplots(3, 4, num = 'Lap Belt Scores %s' % vitesse, 
                         sharex = [1,1,1,1,1,1,1,1,1,1,11,12], 
                         sharey = [1,1,3,3,5,5,3,3,9,10,11,12])
    
    xlabel = 'Lap Belt Score'
    
    for channel, subloc in subplot_peak.items():
        ax = axs[subloc-1]
        
        for tcn in peakdata[channel].index:
            ax.plot(lbs[tcn]['D3'], peakdata[channel].loc[tcn,'peak'], '.',
                     color = 'tab:orange', label = 'C3')
            ax.plot(lbs[tcn]['D4'], peakdata[channel].loc[tcn,'peak'], '.', 
                     color = 'tab:purple', label = 'C4')
            ax.set_title(titlelist[subloc-1]+' Value')
            ylabel = style.ylabel(channel[12:14], channel[14])
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
    
    for channel, subloc in subplot_peaktime.items():
        ax = axs[subloc-1]
        
        for tcn in peakdata[channel].index:
            ax.plot(lbs[tcn]['D3'], peakdata[channel].loc[tcn,'tp'], '.', 
                     color = 'tab:orange', label = 'C3')
            ax.plot(lbs[tcn]['D4'], peakdata[channel].loc[tcn,'tp'], '.', 
                     color = 'tab:purple', label = 'C4')
            ax.set_title(titlelist[subloc-1]+' Time')
            ax.set_ylabel('Time [s]')
            ax.set_xlabel(xlabel)

    ax = axs[9-1]
    for tcn in peakdata['14CHST0000Y7ACXC'].index:
        ax.plot(lbs[tcn]['D3'], 
                peakdata['14CHST0000Y7ACXC'].loc[tcn,'tp'] - 
                peakdata['14PELV0000Y7ACXA'].loc[tcn,'tp'], '.', 
                color='tab:orange', label = 'C3')
        ax.plot(lbs[tcn]['D4'], 
                peakdata['14CHST0000Y7ACXC'].loc[tcn,'tp'] - 
                peakdata['14PELV0000Y7ACXA'].loc[tcn,'tp'], '.', 
                color='tab:purple', label = 'C4')
    ax.set_title(titlelist[9-1])
    ax.set_ylabel('Gap Time [s]')
    ax.set_xlabel(xlabel)
    
    ax = axs[10-1]
    for tcn in peakdata['14CHST0000Y7ACXC'].index:
        ax.plot(lbs[tcn]['D3'], 
                peakdata['14CHST0000Y7ACXC'].loc[tcn,'peak'] - 
                peakdata['14PELV0000Y7ACXA'].loc[tcn,'peak'], '.', 
                color='tab:orange', label='C3')
        ax.plot(lbs[tcn]['D4'], 
                peakdata['14CHST0000Y7ACXC'].loc[tcn,'peak'] - 
                peakdata['14PELV0000Y7ACXA'].loc[tcn,'peak'], '.', 
                color='tab:purple', label='C4')
    ax.set_title(titlelist[10-1])
    ax.set_ylabel('Gap [g]')
    ax.set_xlabel(xlabel)
    
    ax = axs[11-1]
    for tcn in peakdata['14PELV0000Y7ACXA'].index:
        ax.plot(peakdata['14PELV0000Y7ACXA'].loc[tcn,'t0'],
                peakdata['14PELV0000Y7ACXA'].loc[tcn,'tp'], '.',
                color='tab:purple', label='start')
    ax.set_title('Start Time vs Peak Time, Pelvis')
    ax.set_ylabel('Peak Time [s]')
    ax.set_xlabel('Start Time [s]')
    
    ax = axs[12-1]
    for tcn in peakdata['14PELV0000Y7ACXA'].index:
        ax.plot(peakdata['14PELV0000Y7ACXA'].loc[tcn,'t0'],
                peakdata['14PELV0000Y7ACXA'].loc[tcn,'peak'], '.', 
                color='tab:blue', label='start to peak')
    ax.set_title('Start Time vs Peak Pelvis Accel.')
    ax.set_ylabel('Accel [g]')
    ax.set_xlabel('Time [s]')
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.subplots_adjust(top=0.921,bottom=0.059,left=0.05,right=0.99,
                        hspace=0.359,wspace=0.248)

p = peakdata['14PELV0000Y7ACXA'].reset_index()    
p.to_excel('C:/Users/giguerf/peak.xlsx', index=False)
