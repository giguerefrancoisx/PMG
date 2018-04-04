# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:53:00 2018

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
chname = dict(zip(chlist, ['HEAD','HEAD','CHEST','CHEST','PELVIS','PELVIS']))
colors = style.colordict(['old_accel','new_accel','new_decel'])


plt.close('all')
for trio in 'ABCDEFGHIJKLMNOPQRSTU':
    SEs = table[table.TRIO==trio].SE.tolist()
    dummy = table[table.TRIO==trio].DUMMY.tolist()[0]
    model = table[table.TRIO==trio].MODEL.tolist()[0]

    fig, axs = style.subplots(2,2,sharex='all')
    fig.suptitle(model)

    if dummy == 'Y2':
        channels = ['12HEAD0000Y2ACXA','12CHST0000Y2ACXC','12PELV0000Y2ACXA']
    else:
        channels = ['12HEAD0000Y7ACXA','12CHST0000Y7ACXC','12PELV0000Y7ACXA']

    for ch, ax in zip(channels, axs):
        for SE in SEs:
            try:
                sled = table[table.SE==SE].SLED.tolist()[0]
#                ax.plot(time, fulldata1[ch].loc[:,SE], label='_nolegend_', color=colors[sled], alpha=0.2)

                data = fulldata[ch].loc[:,SE].rolling(20,0,center=True,win_type='triang').mean()
                ax.plot(time, data, label=sled+' '+SE, color=colors[sled])
            except KeyError:
                print('Error: {}, {}'.format(SE,ch))
                pass
        ax.set_ylim(-65,75)
        ax.set_title(chname[ch])
    axs[0].set_xlim(0,0.12)
    h, l = axs[0].get_legend_handles_labels()
    axs[3].legend(h, l, loc='upper left')
    plt.tight_layout()
    plt.savefig('P:/SLED/Plots/'+'-'.join([name[:4] for name in model.split(' ')]).upper()+'_'+trio)
    plt.close('all')
#%%
plt.close('all')
for trio in 'EGR':
    SEs = table[table.TRIO==trio].SE.tolist()
    dummy = table[table.TRIO==trio].DUMMY.tolist()[0]
    model = table[table.TRIO==trio].MODEL.tolist()[0]

    fig, axs = style.subplots(2,2,sharex='all')
    fig.suptitle(model)

    if dummy == 'Y2':
        channels = ['12HEAD0000Y2ACXA','12CHST0000Y2ACXC','12PELV0000Y2ACXA']
    else:
        channels = ['12HEAD0000Y7ACXA','12CHST0000Y7ACXC','12PELV0000Y7ACXA']

    for ch, ax in zip(channels, axs):
        for SE in SEs:
            try:
                sled = table[table.SE==SE].SLED.tolist()[0]
#                ax.plot(time, fulldata1[ch].loc[:,SE], label='_nolegend_', color=colors[sled], alpha=0.2)

                data = fulldata[ch].loc[:,SE].rolling(20,0,center=True,win_type='triang').mean()
                ax.plot(time, data, label=sled+' '+SE, color=colors[sled])
            except KeyError:
                print('Error: {}, {}'.format(SE,ch))
                pass
        ax.set_title(chname[ch])
    axs[0].set_xlim(0,0.25)
    h, l = axs[0].get_legend_handles_labels()
    axs[3].legend(h, l, loc='upper left')
    plt.tight_layout()
    plt.savefig('P:/SLED/Plots/'+'-'.join([name[:4] for name in model.split(' ')]).upper()+'_'+trio+'_extra')
plt.close('all')