# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:34:53 2018

@author: giguerf
"""
#import os
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from PMG.COM.openbook import openHDF5
#import PMG.COM.plotstyle as style
#import PMG.COM.table as tb
#
#THOR = os.fspath('P:/AHEC/Data/THOR/')
#chlist = ['11NECKLO00THFOXA', '11CHSTLEUPTHDSXB', '11PELV0000THACXA', '11FEMRLE00THFOZB']
##chlist = ['11CHST0000THACXC', '11CHSTLEUPTHDSXB', '11CLAVLEINTHFOZA', '11CLAVRIINTHFOZA']
#
#time, fulldata = openHDF5(THOR, chlist)
#
#ok, slip = tb.split(tb.get('THOR'), 'CBL_BELT', ['OK','SLIP']).values()
#slip = slip[~slip.T1.isnull() & ~slip.T1.isin(['?'])]
#
#bin1 = slip[(0.060<=slip.T1) & (slip.T1<0.075)]
#bin2 = slip[(0.075<=slip.T1) & (slip.T1<0.090)]
#bin3 = slip[(0.090<=slip.T1) & (slip.T1<0.115)]

#%%
#r, c = 4, 4
#plt.close('all')
#fig, axs = style.subplots(r, c, sharex='all', sharey='col')
#axs = axs.reshape((r,c))
#
#ind = pd.concat([time, pd.Series(time.index)], axis=1).set_index('Time')
#
#for c, ch in enumerate(chlist[:4]):
#    for r, group in enumerate([bin1, bin2, bin3]):
#
#        ax = axs[r,c]
#        tcns = group.CIBLE.tolist()
#        points = np.array([])
#        values = np.array([])
#        points2 = np.array([])
#        values2 = np.array([])
#        for tcn in tcns:
#            point = group[group.CIBLE == tcn].T1.iloc[0]
#            point2 = group[group.CIBLE == tcn].T2.iloc[0]
#            index = ind.loc[point].iloc[0]
#            index2 = ind.loc[point2].iloc[0]
#            try:
#                value = fulldata[ch].loc[index, tcn]
#                value2 = fulldata[ch].loc[index2, tcn]
#            except KeyError:
#                value = np.nan
#                value2 = np.nan
#            points = np.append(points, point)
#            values = np.append(values, value)
#            points2 = np.append(points2, point2)
#            values2 = np.append(values2, value2)
#
#        df = fulldata[ch].loc[:,tcns]
#        ax.plot(time, df)
#        ax.plot(points, values, '.', color='b')
#        ax.plot(points2, values2, '.', color='r')
#        ax.axvline(0.060, color='k', alpha = 0.5)
#        ax.axvline(0.115, color='k', alpha = 0.5)
#
#        for line, tcn in zip(ax.lines, df.columns):
#            line.set_label(tcn)
#
#    axs[0,0].set_ylabel('Early Slip')
#    axs[1,0].set_ylabel('Late Slip')
#    axs[2,0].set_ylabel('Very Late Slip')
#
#    r = r+1
#    group = ok
#    ax = axs[r,c]
#    tcns = group.CIBLE.tolist()
#    df = fulldata[ch].loc[:,tcns]
#    ax.plot(time, df)
#
#    for line, tcn in zip(ax.lines, df.columns):
#            line.set_label(tcn)
#
#    axs[r,0].set_ylabel('OK Belts')
#
#    axs[0,c].set_title(ch)
#ax.set_xlim(0,0.2)
#
#for c, _ in enumerate(chlist[:4]):
#    for r, (left, right) in enumerate(zip([0.06,0.075,0.09],[0.075, 0.09, 0.115])):
#        ax = axs[r,c]
#        bot, top = ax.get_ylim()
#        ax.fill_between(np.linspace(left, right, 10), bot*np.ones((10,)), top*np.ones((10,)), color = 'tab:blue', alpha = 0.25)
#        ax.set_ylim(bot, top)
#
#style.maximize()
#%%
#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from PMG.COM.openbook import openHDF5
import PMG.COM.plotstyle as style
import PMG.COM.table as tb

#THOR = os.fspath('P:/AHEC/Data/THOR/')
chlist = ['11NECKLO00THFOXA', '11CHSTLEUPTHDSXB', '11PELV0000THACXA', '11FEMRLE00THFOZB']
#chlist = ['11CHST0000THACXC', '11CHSTLEUPTHDSXB', '11CLAVLEINTHFOZA', '11CLAVRIINTHFOZA']

time, fulldata = openHDF5(THOR, chlist)

ok, slip = tb.split(tb.get('THOR'), 'CBL_BELT', ['OK','SLIP']).values()
slip = slip[~slip.T1.isnull() & ~slip.T1.isin(['?'])]

bin1 = slip[(0.060<=slip.T1) & (slip.T1<0.075)]
#bin2 = slip[(0.075<=slip.T1) & (slip.T1<0.090)]
#bin3 = slip[(0.090<=slip.T1) & (slip.T1<0.115)]
#%%
ok, slip = tb.split(table, 'CBL_BELT', ['OK','SLIP']).values()
slip = slip[~slip.T1.isnull() & ~slip.T1.isin(['?'])]
bin1 = slip[(0.060<=slip.T1) & (slip.T1<0.075)]

r, c = 2, 2
plt.close('all')
fig, axs = style.subplots(r, c, sharex='all', sharey='col', figsize=(6.5, 6.5))
#axs = axs.reshape((r,c))

ind = pd.concat([time, pd.Series(time.index)], axis=1).set_index('Time')
chname = dict(zip(chlist, ['Lower Neck $F_x$', 'Upper Left Chest $D_x$']))

for c, (ch,group) in enumerate(zip(['11NECKLO00THFOXA', '11CHSTLEUPTHDSXB','11NECKLO00THFOXA', '11CHSTLEUPTHDSXB'],[bin1,bin1,ok,ok])):

    ax=axs[c]
    tcns = group.CIBLE.tolist()
    df = fulldata[ch].loc[:,tcns]
    if c in [0,2]:
        colors = style.colordict(fulldata[ch], 'max', plt.cm.viridis)
#        vmin, vmax = fulldata[ch].max().min(), fulldata[ch].max().max()
#        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=plt.Normalize(vmin, vmax))
    if c in [1,3]:
        colors = style.colordict(fulldata[ch], 'min', plt.cm.viridis_r)
#        vmin, vmax = fulldata[ch].min().min(), fulldata[ch].min().max()
#        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=plt.Normalize(vmin, vmax))

    for tcn in df.columns:
        ax.plot(time, df[tcn], color=colors[tcn])

    if c in [0,1]:
        points = np.array([])
        values = np.array([])
        points2 = np.array([])
        values2 = np.array([])
        for tcn in tcns:
            point = group[group.CIBLE == tcn].T1.iloc[0]
            point2 = group[group.CIBLE == tcn].T2.iloc[0]
            index = ind.loc[point].iloc[0]
            index2 = ind.loc[point2].iloc[0]
            try:
                value = fulldata[ch].loc[index, tcn]
                value2 = fulldata[ch].loc[index2, tcn]
            except KeyError:
                value = np.nan
                value2 = np.nan
            points = np.append(points, point)
            values = np.append(values, value)
            points2 = np.append(points2, point2)
            values2 = np.append(values2, value2)
        ax.plot(points, values, '.', color='b')
        ax.plot(points2, values2, '.', color='r')
        axs[c].set_title(chname[ch])
    if c in [2,3]:
        axs[c].set_xlabel('Time [s]')
#        sm._A = []
#        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal')

axs[0].set_ylabel('Early Slip')
axs[2].set_ylabel('No-Slip Belts')

ax.set_xlim(0,0.2)
