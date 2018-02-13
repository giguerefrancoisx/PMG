# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:34:53 2018

@author: giguerf
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PMG.COM.openbook import openHDF5
import PMG.COM.plotstyle as style

THOR = os.fspath('P:/AHEC/Data/THOR/')
chlist = ['11NECKLO00THFOXA','11CHSTLEUPTHDSXB']

time, fulldata = openHDF5(THOR, chlist)

table = pd.read_excel('P:/AHEC/thortable.xlsx', sheetname='All')
table = table.dropna(axis=0, thresh=5).dropna(axis=1, how='all')

ok = table[table.CBL_BELT.isin(['OK'])]
slip = table[table.CBL_BELT.isin(['SLIP']) & ~table.T1.isnull() & ~table.T1.isin(['?'])]

bin1 = slip[(0.060<=slip.T1) & (slip.T1<0.075)]

#%%
from collections import OrderedDict
plt.close('all')
fig, axs = style.subplots(2,2,sharex='all',sharey='col')

ind = pd.concat([time, pd.Series(time.index)], axis=1).set_index('Time')

ch = chlist[0]

## Slips
ax = axs[0]
tcns = bin1.CIBLE.tolist()[:10]
points = np.array([])
values = np.array([])
for tcn in tcns:
    point = bin1[bin1.CIBLE == tcn].T1.iloc[0]
    index = ind.loc[point].iloc[0]
    try:
        value = fulldata[ch].loc[index, tcn]
    except KeyError:
        value = np.nan
    points = np.append(points, point)
    values = np.append(values, value)

df = fulldata[ch].loc[:,tcns]
ax.plot(time, df)
ax.plot(points, values, '.', color='b', label='Belt begins slipping')
ax.axvline(0.060, color='k', alpha = 0.5)
ax.axvline(0.075, color='k', alpha = 0.5)

for line, tcn in zip(ax.lines, df.columns):
    line.set_label(tcn)

ax.set_title('Lower Neck Force in X [N]')
ax.set_ylabel('Slipping Belts')

handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels[-1:], handles[-1:]))
ax.legend(by_label.values(), by_label.keys(), loc=4)

## OKs
ax = axs[2]
tcns = ok.CIBLE.tolist()[-10:]
df = fulldata[ch].loc[:,tcns]
ax.plot(time, df)

for line, tcn in zip(ax.lines, df.columns):
    line.set_label(tcn)

ax.set_ylabel('Stationary Belts')
ax.set_xlabel('Time [s]')
ax.set_xlim(0,0.2)


### Chest
ch = chlist[1]

## Slips
ax = axs[1]
tcns = bin1.CIBLE.tolist()[:10]
points = np.array([])
values = np.array([])
for tcn in tcns:
    point = bin1[bin1.CIBLE == tcn].T1.iloc[0]
    index = ind.loc[point].iloc[0]
    try:
        value = fulldata[ch].loc[index, tcn]
    except KeyError:
        value = np.nan
    points = np.append(points, point)
    values = np.append(values, value)

df = fulldata[ch].loc[:,tcns]
ax.plot(time, df)
ax.plot(points, values, '.', color='b', label='Belt begins slipping')
ax.axvline(0.060, color='k', alpha = 0.5)
ax.axvline(0.075, color='k', alpha = 0.5)

for line, tcn in zip(ax.lines, df.columns):
    line.set_label(tcn)

ax.set_title('Upper Left Chest Displacement [mm]')
ax.set_ylabel('Slipping Belts')

handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels[-1:], handles[-1:]))
ax.legend(by_label.values(), by_label.keys(), loc=4)

## OKs
ax = axs[3]
tcns = ok.CIBLE.tolist()[-10:]
df = fulldata[ch].loc[:,tcns]
ax.plot(time, df)

for line, tcn in zip(ax.lines, df.columns):
    line.set_label(tcn)

ax.set_ylabel('Stationary Belts')
ax.set_xlabel('Time [s]')
ax.set_xlim(0,0.2)
