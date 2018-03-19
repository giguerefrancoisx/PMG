# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:23:40 2018

@author: giguerf
"""
import matplotlib.pyplot as plt
import numpy as np
from PMG.COM import data, plotstyle as style, table as tb

THOR = 'P:/AHEC/Data/THOR/'
chlist = ['11NECKLO00THFOXA', '11CHSTLELOTHDSXB','11CHST0000THACXC','11CHSTRILOTHDSXB','11CHSTRIUPTHDSXB','11SPIN0100THACYC','11SEBE0000B3FO0D']
time, fulldata = data.import_data(THOR, chlist, check=False)

table = tb.get('THOR')
table = table[table.TYPE.isin(['Frontale/Véhicule'])]
slips  = table[table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
oks = table[table.CBL_BELT.isin(['OK'])].CIBLE.tolist()

#%%
my_color = 'tab:blue'
plt.close('all')
#plt.figure()
df = fulldata['11CHSTRILOTHDSXB'].loc[:,:].dropna(axis=1)
#for tcn in df.columns:
#    plt.plot(time, df[tcn], '.', color='tab:blue', markersize=0.5, alpha=1)

fig, axs = plt.subplots(1,2,sharex=True,sharey=True, figsize=(12,6))
ax = axs[0]
ax.plot(time, df, color=my_color, lw=1, alpha=1)

ax = axs[1]
ax.plot(time, df.median(axis=1).rolling(100, 0, center=True, win_type='parzen').mean(), color=my_color)
for alpha in [0.5,0.2,0.1,0.05,0.025, 0]:
    ax.fill_between(time, df.quantile(alpha/2, axis=1).rolling(100, 0, center=True, win_type='parzen').mean(),
                     df.quantile(1-alpha/2, axis=1).rolling(100, 0, center=True, win_type='parzen').mean(),
                     color=my_color, alpha=0.2)
ax.set_xlim(0,0.3)
plt.tight_layout()
#%%
my_color = 'tab:blue'
plt.close('all')
df = fulldata['11SPIN0100THACYC'].loc[:,:].dropna(axis=1)

N = 10
band = 0.3 #0.2-0.5 max quantile
#quantiles = np.power(2.0, (np.arange(0,N+1,1)-N))*band; quantiles[0]=0
#quantiles = np.arange(0,N+1,1)/N*band
quantiles = np.arange(0,N+1,1)/20

#colors = style.colordict(quantiles, 'order', ['tab:red','tab:blue'], N+1)

fig, axs = plt.subplots(1,2,sharex=True,sharey=True, figsize=(12,4))
ax = axs[0]
ax.plot(time, df, color=my_color, lw=1, alpha=1)

ax = axs[1]
ax.plot(time, df.median(axis=1).rolling(100, 0, center=True, win_type='parzen').mean(), color=my_color)
for i, alpha in enumerate(quantiles):
    ax.fill_between(time, df.quantile(alpha/2, axis=1).rolling(100, 0, center=True, win_type='parzen').mean(),
                     df.quantile(1-alpha/2, axis=1).rolling(100, 0, center=True, win_type='parzen').mean(),
                     alpha=0.10, color=my_color, lw=0)#colors[i])
ax.set_xlim(0,0.3)
plt.tight_layout()
