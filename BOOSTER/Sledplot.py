# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:46:55 2017

@author: giguerf
"""
import pandas as pd
import matplotlib.pyplot as plt
from GitHub.COM import plotstyle as style

MifoldData = pd.read_excel('P:/BOOSTER/DATA/SE18-0009(SAI).xls', 
                           header = 0, index_col = 0, skiprows = [1,2])
TakataData = pd.read_excel('P:/BOOSTER/DATA/SE18-0007_4(SAI).xls', 
                           header = 0, index_col = 0, skiprows = [1,2])

#data = dict(zip(['TC18-011','TC18-018'],[MifoldData,TakataData]))
time = MifoldData['T_10000_0']
channels = ['12CHST0000Y7ACXC','12PELV0000Y7ACXA']
colors = dict(zip(channels, ['tab:orange','tab:blue']))
name = dict(zip(channels, ['Chest','Pelvis']))
#%%
plt.close('all')
#plt.figure(figsize=(4.8,4.8))
#ax1 = plt.subplot(211)
#ax2 = plt.subplot(212)
axs = style.subplots(2,1,sharex='all',sharey='all',visible=True,figsize=(4.8,4.8))
ax1,ax2 = axs.flatten()

for ch in channels:
    ax1.plot(time*1000, MifoldData[ch], color = colors[ch], label = name[ch])
    ax2.plot(time*1000, TakataData[ch], color = colors[ch], label = name[ch])
    
for ax in [ax1,ax2]:
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Acceleration X [g]')
    ax.set_ylim((-80,15))
    ax.set_xlim((0,200))
    
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc=4)

#ax1.set_title('Mifold MF01-US')
#ax2.set_title('Takata 312-Neo Junior')


plt.tight_layout()
