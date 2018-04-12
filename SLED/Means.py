# -*- coding: utf-8 -*-
"""
Created on Th Mar 29 2108

@author: giguerf
"""
import matplotlib.pyplot as plt
from PMG.COM.openbook import openHDF5
from PMG.COM import table as tb, plotstyle as style

table = tb.get('SLED')

SLED = 'P:/SLED/Data/'
chlist = ['12HEAD0000Y7ACXA','12HEAD0000Y2ACXA',
          '12CHST0000Y7ACXC','12CHST0000Y2ACXC',
          '12PELV0000Y7ACXA','12PELV0000Y2ACXA']

time, fulldata = openHDF5(SLED, chlist)

#%%
table = table[table.BACK.isin(['HB', 'LB']) & table.DATA.isin(['YES'])]
colors = style.colordict(['HBold_accel', 'LBold_accel','HBnew_accel', 'LBnew_accel'])
labels = dict(zip(['HBold_accel', 'LBold_accel','HBnew_accel', 'LBnew_accel'],
                  ['Highback Old', 'Lowback Old','Highback New', 'Lowback New']))

channels = ['12HEAD0000Y7ACXA','12CHST0000Y7ACXC','12PELV0000Y7ACXA']
chname = dict(zip(channels, ['Head','Chest','Pelvis']))

plt.close('all')
for ch in channels:

    fig, axs = style.subplots(2,2, sharex='all', sharey='all')
    for i, (back, sled, ax) in enumerate(zip(['HB', 'HB','LB', 'LB'], ['old_accel','new_accel','old_accel','new_accel'], axs)):
        SEs = table[table.BACK.isin([back]) & table.SLED.isin([sled])].SE.tolist()
        data = fulldata[ch].loc[:,SEs].rolling(20,0,center=True,win_type='triang').mean()
        ax.plot(time, data, color=colors[back+sled], label=labels[back+sled])
        style.legend(ax, loc='lower right')
    ax.set_xlim(0,0.12)
    ax.set_ylim(-65,15)
    axs[0].set_ylabel('Highback')
    axs[2].set_ylabel('Lowback')
    axs[0].set_title('Old Sled, '+chname[ch])
    axs[1].set_title('New Sled, '+chname[ch])
    axs[2].set_xlabel('Time [s]')
    axs[3].set_xlabel('Time [s]')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig('P:/SLED/Plots/Grid_All_'+chname[ch])
#%%
plt.close('all')
labels = dict(zip(['old_accel', 'new_accel'], ['Old Sled','New Sled']))
fig, axs = style.subplots(2,2, sharex='all', sharey='all')
axs = axs.reshape(2,2)
for j, ch in enumerate(['12CHST0000Y7ACXC','12PELV0000Y7ACXA']):
    axs[0,j].set_title(chname[ch])
    for i, (back, ax) in enumerate(zip(['HB', 'LB'], axs[:,j])):
        for sled in ['old_accel','new_accel']:
            SEs = table[table.BACK.isin([back]) & table.SLED.isin([sled])].SE.tolist()
            data = fulldata[ch].loc[:,SEs].rolling(20,0,center=True,win_type='triang').mean()
            ax.plot(time, data, color=colors[back+sled], label=labels[sled]+', n={}'.format(data.shape[1]))
        style.legend(ax, loc='lower right')
axs[0,0].set_ylabel('Highback')
axs[1,0].set_ylabel('Lowback')
axs[1,0].set_xlabel('Time [s]')
axs[1,1].set_xlabel('Time [s]')
axs[-1,-1].set_xlim(0,0.12)
axs[-1,-1].set_ylim(-65,15)
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('P:/SLED/Plots/Grid_All_Chest_Pelvis')
#%%
plt.close('all')
for ch in channels:

    fig, axs = style.subplots(2,2, sharex='all', sharey='all')
    for i, (back, sled, ax) in enumerate(zip(['HB', 'HB','LB', 'LB'], ['old_accel','new_accel','old_accel','new_accel'], axs)):
        SEs = table[table.BACK.isin([back]) & table.SLED.isin([sled])].SE.tolist()
        data = fulldata[ch].loc[:,SEs].rolling(20,0,center=True,win_type='triang').mean()
        median = fulldata[ch].loc[:,SEs].median(axis=1).rolling(20,0,center=True,win_type='triang').mean()
        ax.plot(time, median, color=colors[back+sled], label='Median, n={}'.format(data.shape[1]))
        ax.fill_between(time, data.min(axis=1), data.max(axis=1), color=colors[back+sled], label='Range', alpha=0.2)
        style.legend(ax, loc='lower right')
    ax.set_xlim(0,0.12)
    ax.set_ylim(-65,15)
    axs[0].set_ylabel('Highback')
    axs[2].set_ylabel('Lowback')
    axs[0].set_title('Old Sled, '+chname[ch])
    axs[1].set_title('New Sled, '+chname[ch])
    axs[2].set_xlabel('Time [s]')
    axs[3].set_xlabel('Time [s]')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig('P:/SLED/Plots/Grid_Means_'+chname[ch])
    plt.close('all')
#%%
plt.close('all')
labels = dict(zip(['old_accel', 'new_accel'], ['Old Sled','New Sled']))
fig, axs = style.subplots(2,2, sharex='all', sharey='all')
axs = axs.reshape(2,2)
for j, ch in enumerate(['12CHST0000Y7ACXC','12PELV0000Y7ACXA']):
    axs[0,j].set_title(chname[ch])
    for i, (back, ax) in enumerate(zip(['HB', 'LB'], axs[:,j])):
        for sled in ['old_accel','new_accel']:
            SEs = table[table.BACK.isin([back]) & table.SLED.isin([sled])].SE.tolist()
            data = fulldata[ch].loc[:,SEs].rolling(20,0,center=True,win_type='triang').mean()
            median = fulldata[ch].loc[:,SEs].median(axis=1).rolling(20,0,center=True,win_type='triang').mean()
            ax.plot(time, median, color=colors[back+sled], label=labels[sled]+', n={}'.format(data.shape[1]))
            ax.fill_between(time, data.min(axis=1), data.max(axis=1), color=colors[back+sled], label=labels[sled], alpha=0.2)
        ax.legend(loc='lower right')
axs[0,0].set_ylabel('Highback')
axs[1,0].set_ylabel('Lowback')
axs[1,0].set_xlabel('Time [s]')
axs[1,1].set_xlabel('Time [s]')
axs[-1,-1].set_xlim(0,0.12)
axs[-1,-1].set_ylim(-65,15)
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('P:/SLED/Plots/Grid_Means_Chest_Pelvis')

plt.close('all')

