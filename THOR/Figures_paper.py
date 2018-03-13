# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:51:04 2018

@author: giguerf
"""
import matplotlib
import matplotlib.pyplot as plt
import PMG.COM.data as dat
import PMG.COM.table as tb
import PMG.COM.plotstyle as style
THOR = 'P:/AHEC/Data/THOR/'
chlist = ['11NECKLO00THFOXA','11NECKLO00THFOYA','11CHSTLEUPTHDSXB','11FEMRLE00THFOZB']
time, fulldata = dat.import_data(THOR, chlist)
df = fulldata['11NECKLO00THFOXA'].dropna(axis=1)
table = tb.get('THOR')
table = table[table.TYPE.isin(['Frontale/Véhicule'])]
slips  = table[table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
oks = table[table.CBL_BELT.isin(['OK'])].CIBLE.tolist()

#%% Comparison of slip and ok traces for select channels
#plt.close('all')
#df = df.loc[:,slips+oks]
#colordf = df.copy()
#colordf['slip_median'] = df.loc[:,slips].median(axis=1)
#colordf['ok_median'] = df.loc[:,oks].median(axis=1)
#
#colors = style.colordict(colordf, by='min', values=plt.cm.rainbow)
#
#fig, axs = style.subplots(2,2, sharex='all', sharey='all', figsize=(6.5,4))
#for tcn in slips:
#    axs[0].plot(time, df.loc[:,tcn], color=colors[tcn])
#for tcn in oks:
#    axs[1].plot(time, df.loc[:,tcn], color=colors[tcn])
#
#window = 100
#slip_median = df.loc[:,slips].median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
#ok_median = df.loc[:,oks].median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
#slip_high = df.loc[:,slips].quantile(0.85, axis=1).rolling(window,0,center=True,win_type='triang').mean()
#slip_low = df.loc[:,slips].quantile(0.15, axis=1).rolling(window,0,center=True,win_type='triang').mean()
#ok_high = df.loc[:,oks].quantile(0.85, axis=1).rolling(window,0,center=True,win_type='triang').mean()
#ok_low = df.loc[:,oks].quantile(0.15, axis=1).rolling(window,0,center=True,win_type='triang').mean()
#
#
#axs[2].plot(time, slip_median, color=colors['slip_median'], label='Median, n={}'.format(len(slips)))
#axs[2].plot(time, ok_median, color=colors['ok_median'], label='Median, n={}'.format(len(oks)))
#axs[2].fill_between(time, slip_high, slip_low, color=colors['slip_median'], alpha=0.2, label='5th-95th Quantiles')
#axs[2].fill_between(time, ok_high, ok_low, color=colors['ok_median'], alpha=0.2, label='5th-95th Quantiles')
##axs[2].legend()
#
#axs[0].set_xlim(0,0.3)
##axs[0].set_ylim()
#axs[2].set_xlabel('Time [s]')
#axs[0].set_ylabel('Lower Neck $F_x$ [N]')
#%% Comparison of slip and ok traces for select channels
plt.close('all')
#chlist = ['11NECKLO00THFOXA','11NECKLO00THFOYA']
labels = ['Lower Neck $\mathregular{F_x}$ [N]',
          'Lower Neck $\mathregular{F_y}$ [N]',
          'Upper Left Chest $\mathregular{D_x}$ [mm]',
          'Left Femur $\mathregular{F_x}$ [N]']
ylabel = dict(zip(chlist, labels))
xfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
xfmt.set_powerlimits((-3,4))

slip_color = 'tab:blue'
ok_color = 'tab:red'
n = len(chlist)

fig, axs = style.subplots(n, 2, sharex='all', sharey='row', figsize=(6.5,2*n))
plt.rcParams['font.size']= 10

for i, channel in enumerate(chlist):
    df = fulldata[channel].dropna(axis=1)
    df = df.loc[:,slips+oks]
    for tcn in slips:
        axs[0+2*i].plot(time, df.loc[:,tcn], color=slip_color, lw=1, label='Slip')
    for tcn in oks:
        axs[0+2*i].plot(time, df.loc[:,tcn], color=ok_color, lw=1, label='No-Slip')

    window = 100
    alpha = 0.10
    slip_median = df.loc[:,slips].median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ok_median = df.loc[:,oks].median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
    slip_high = df.loc[:,slips].quantile(1-alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
    slip_low = df.loc[:,slips].quantile(alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ok_high = df.loc[:,oks].quantile(1-alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ok_low = df.loc[:,oks].quantile(alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()


    axs[1+2*i].plot(time, slip_median, color=slip_color, label='Median, n={}'.format(len(slips)))
    axs[1+2*i].plot(time, ok_median, color=ok_color, label='Median, n={}'.format(len(oks)))
    axs[1+2*i].fill_between(time, slip_high, slip_low, color=slip_color, alpha=0.2, label='{:2.0f}th Percentile'.format(100*(1-alpha)))
    axs[1+2*i].fill_between(time, ok_high, ok_low, color=ok_color, alpha=0.2, label='{:2.0f}th Percentile'.format(100*(1-alpha)))
#    axs[1+2*i].legend(loc='lower right', fontsize=6)

    axs[0+2*i].set_ylabel(ylabel[channel])
    axs[0+2*i].yaxis.set_label_coords(-0.28,0.5)
    axs[0+2*i].yaxis.set_major_formatter(xfmt)

axs[-1].set_xlim(0,0.3)
axs[-1].set_xlabel('Time [s]')
axs[-2].set_xlabel('Time [s]')

style.legend(axs[-2], loc='lower right', fontsize=6)
axs[-1].legend(loc='lower right', fontsize=6)

plt.tight_layout()
#%% Figure
