# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:46:55 2017

@author: giguerf
"""
import os
import sys
#import pandas as pd
import matplotlib.pyplot as plt
if 'C:/Users/giguerf/Documents' not in sys.path:
    sys.path.insert(0, 'C:/Users/giguerf/Documents')
from GitHub.COM import plotstyle as style
from GitHub.COM import openbook as ob
#from GitHub.COM import get_peak as gp

readdir = os.fspath('P:/BOOSTER/SAI/Y7')
channels = ['14CHST0000Y7ACXC','14PELV0000Y7ACXA']

time, fulldata, *_ = ob.openbook(readdir)
groupdict = ob.popgrids(fulldata, channels)

#for group in groupdict:
#    for channel in groupdict[group]:
#        temp = style.explode(groupdict[group][channel],[])
#        dvdf = pd.concat(temp, axis=1)
#        groupdict[group][channel] = dvdf

#groupA = groupdict['A']
#groupC = groupdict['C']
chm = [channel+'m' for channel in channels]
colors = dict(zip(channels+chm, ['tab:orange','tab:blue','tab:red','tab:blue']))
name = dict(zip(channels, ['Chest','Pelvis']))
al = dict(zip(channels, [0.25,0.25]))
#%% Group A and C for Offset, Wall
plt.close('all')

axs = style.subplots(2,2,sharex='all',sharey='all',visible=True,figsize=(9,6))

for i, config in enumerate(['offset','mur']):
    
    for ch in channels:
        axs[i].plot(time*1000, groupdict['A'][ch][config], color = colors[ch], alpha=al[ch], label = name[ch])
        axs[i+2].plot(time*1000, groupdict['C'][ch][config], color = colors[ch], alpha=al[ch], label = name[ch])
        
        axs[i].plot(time*1000, groupdict['A'][ch][config].mean(axis=1), color = colors[ch+'m'], label = name[ch]+' Mean')
        axs[i+2].plot(time*1000, groupdict['C'][ch][config].mean(axis=1), color = colors[ch+'m'], label = name[ch]+' Mean')
        
    for group in ['A','C']:
        ax = axs[i] if group == 'A' else axs[i+2]
        
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Acceleration X [g]')
        lim = (-140,40)
        ax.set_ylim(lim)
        ax.set_xlim((0,200))
        annotation = groupdict[group][ch][config].shape[1]
        ax.annotate('n = %d' % annotation, (0.01, 0.01), xycoords='axes fraction')
    
    #    h, l = ax.get_legend_handles_labels()
    #    ax.legend(h, l, loc=4)
        style.legend(ax=ax, loc=4)
        
    plt.yticks(range(*(-140,41,20)))
    
    title = 'Offset' if config == 'offset' else 'FFRB'
    axs[i].set_title('Group \'A\' {}'.format(title))
    axs[i+2].set_title('Group \'B\' {}'.format(title))


plt.tight_layout()

#plt.savefig('P:/BOOSTER/Plots/GroupAC_Chest_Pelvis2.png', dpi=300)
#%% Group A and C for Prius
#plt.close('all')

al = dict(zip(channels, [0.75,0.75]))
axs = style.subplots(1,2,sharex='all',sharey='all',visible=True,figsize=(9,3))
config = 'prius'

for ch in channels:
    axs[0].plot(time*1000, groupdict['A'][ch][config], color = colors[ch], alpha=al[ch], label = name[ch])
    axs[1].plot(time*1000, groupdict['C'][ch][config], color = colors[ch], alpha=al[ch], label = name[ch])
    
#    axs[0].plot(time*1000, groupdict['A'][ch][config].mean(axis=1), color = colors[ch+'m'], label = name[ch]+' Mean')
#    axs[1].plot(time*1000, groupdict['C'][ch][config].mean(axis=1), color = colors[ch+'m'], label = name[ch]+' Mean')
    
for group in ['A','C']:
    ax = axs[0] if group == 'A' else axs[1]
    
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Acceleration X [g]')
    lim = (-140,40)
    ax.set_ylim(lim)
    ax.set_xlim((0,200))
    annotation = groupdict[group][ch][config].shape[1]
    ax.annotate('n = %d' % annotation, (0.01, 0.01), xycoords='axes fraction')

#    h, l = ax.get_legend_handles_labels()
#    ax.legend(h, l, loc=4)
    style.legend(ax=ax, loc=4)
    
plt.yticks(range(*(-140,41,20)))

title = 'Vehicle Buck'
axs[0].set_title('Group \'A\' {}'.format(title))
axs[1].set_title('Group \'B\' {}'.format(title))


plt.tight_layout()
