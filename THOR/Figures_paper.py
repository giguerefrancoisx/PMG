# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:51:04 2018

@author: giguerf
"""
import numpy as np
#import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import PMG.COM.data as dat
import PMG.COM.table as tb
import PMG.COM.plotstyle as style
THOR = 'P:/AHEC/Data/THOR/'
chlist = []
chlist.extend(['11NECKLO00THFOXA','11NECKLO00THFOYA','11CHSTLEUPTHDSXB','11FEMRLE00THFOZB'])
chlist.extend(['11HEAD0000THACXA','11SPIN0100THACXC', '11CHST0000THACXC', '11SPIN1200THACXC', '11PELV0000THACXA'])
chlist.extend(['11HEAD0000THACYA','11SPIN0100THACYC', '11CHST0000THACYC', '11SPIN1200THACYC', '11PELV0000THACYA'])
time, fulldata = dat.import_data(THOR, chlist, check=False)
table = tb.get('THOR')
table = table[table.TYPE.isin(['Frontale/Véhicule'])]
slips  = table[table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
oks = table[table.CBL_BELT.isin(['OK'])].CIBLE.tolist()

plt.rcParams['font.size']= 10
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
if 0:
    plt.close('all')
    chlist = ['11NECKLO00THFOXA','11NECKLO00THFOYA','11CHSTLEUPTHDSXB','11FEMRLE00THFOZB']
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
#%% Bootstrapped timing differences - Redo with
def bootstrap_resample(X, n=1):
    X_resample = np.zeros((n,len(X)))
    for i in range(n):
        resample_i = np.floor(np.random.rand(len(X))*len(X)).astype(int)
        X_resample[i] = X[resample_i]
    return X_resample

if 0:
    chlist = ['11HEAD0000THACXA','11SPIN0100THACXC', '11CHST0000THACXC', '11SPIN1200THACXC', '11PELV0000THACXA']
    #chlist = ['11HEAD0000THACYA','11SPIN0100THACYC', '11CHST0000THACYC', '11SPIN1200THACYC', '11PELV0000THACYA']
    chname = ['Head', 'Spine T1', 'Chest', 'Spine T12', 'Pelvis']

    slip_color = 'tab:blue'
    ok_color = 'tab:red'
###
    #ch = chlist[0]
    #df = fulldata[ch].dropna(axis=1).iloc[:,:20]
    #peaks, times = dat.find_peak(dat.smooth_peaks(df), time)
    #sm = dat.smooth_peaks(df)
    #peaks, times = dat.find_peak(sm, time)
    #plt.figure()
    #plt.plot(time, df, alpha=0.2)
    #plt.plot(time, sm)#, alpha=0.2)
    #plt.plot(times, peaks, '.', color='k')

###
    #plt.close('all')
    #fig, axs = style.subplots(2,2,sharex='all', sharey='all')
    #for ax, n in zip(axs, [500,1000,5000,10000]):
    #    x1_r = bootstrap_resample(times.values, n=n)
    #    means = x1_r.mean(axis=1)
    #    heights, bin_edges = np.histogram(means, bins=np.linspace(means.min(), means.max(),100), density=True)
    #    ax.bar(bin_edges[:-1], heights, width=np.diff(bin_edges)[0])
###
#    ch = chlist[0]
#
#    plt.close('all')
#    plt.figure()
#    ax = plt.gca()
#    for group, label in zip([slips, oks], ['Slip', 'No-Slip']):
#        df = fulldata[ch].dropna(axis=1).loc[:,group].dropna(axis=1)#.iloc[:,:20]
#        peaks, times = dat.find_peak(dat.smooth_peaks(df), time)
#        x1_r = bootstrap_resample(times.values, n=5000)
#        means = x1_r.mean(axis=1)
#        heights, bin_edges = np.histogram(means, bins=np.linspace(means.min(), means.max(),100), density=True)
#        ax.bar(bin_edges[:-1], heights, width=np.diff(bin_edges)[0], label=label, alpha=0.5)
#        ax.legend()
#
#    fig, axs = style.subplots(1, 2, sharex='all', sharey='all')
#    for group, ax in zip([slips, oks], axs):
#        df = fulldata[ch].dropna(axis=1).loc[:,group].dropna(axis=1)#.iloc[:,:20]
#        sm = dat.smooth_peaks(df)
#        peaks, times = dat.find_peak(sm, time)
#        ax.plot(time, sm)
#        ax.plot(times, peaks, '.', color='k')

###
    allpeaks = {}
    alltimes = {}
    for ch in chlist:
        for group, label in zip([slips, oks], ['Slip', 'No-Slip']):
            df = fulldata[ch].dropna(axis=1).loc[:,group].dropna(axis=1)#.iloc[:,:20]
#            sm = dat.smooth_peaks(df)
            sm = df
            allpeaks[ch+label], alltimes[ch+label] = dat.find_peak(sm, time)

    colors = dict(zip(['Slip', 'No-Slip'], [slip_color, ok_color]))

    plt.close('all')
    fig, axs = style.subplots(len(chlist), 1, sharex='all', sharey='all', figsize=(5,5), visible=False)

    for i, (ch, name) in enumerate(zip(chlist, chname)):

        for group, label in zip([slips, oks], ['Slip', 'No-Slip']):
            df = fulldata[ch].dropna(axis=1).loc[:,group].dropna(axis=1)#.iloc[:,:20]
            peaks, times = allpeaks[ch+label], alltimes[ch+label]
            x1_r = bootstrap_resample(times.values, n=5000)
            means = x1_r.mean(axis=1)
#            heights, bin_edges = np.histogram(means, bins=np.linspace(means.min(), means.max(),100), density=True)
            heights, bin_edges = np.histogram(means, bins=np.linspace(0.065, 0.105, 80), density=True)
            axs[i].bar(bin_edges[:-1], heights, width=np.diff(bin_edges)[0], color=colors[label], label=label, alpha=0.5)
#            axs[i].text(0.01, 0.85, name, horizontalalignment='left', verticalalignment='center', transform=axs[i].transAxes)

    axs[-1].set_ylim(10,400)
    axs[-1].set_xlim(0.065,0.105)
    axs[-1].set_xlabel('Times [s]')
    axs[-1].legend(loc='lower right')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

### Convert to Ridgeline Plot
    overlap = 0.50
    for ax, name in zip(axs, chname):
#        ax.text(-0.01, 0.75-overlap, name, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        ax.text(-0.01, 0.5*(1-overlap), name, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        ax.patch.set_alpha(0)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_edgecolor((0.85,0.85,0.85))
        ax.tick_params(left='off', labelleft='off', bottom='off')

    if overlap==0.5:
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['left'].set_visible(False)
        axs[1].spines['top'].set_visible(True)
    else:
        axs[0].spines['top'].set_visible(True)

    axs[-1].tick_params(bottom='on')
    axs[-1].spines['bottom'].set_edgecolor((0,0,0))

    plt.subplots_adjust(left=0.177, top=0.97, hspace=-overlap)
#%% SET UP SB FRACTION PLOTS
columns = ['CIBLE','CBL_BELT','T1','T2','FRACTION']
SB_table = table.loc[table.FRACTION.dropna().index,columns].set_index('CIBLE')
#fraction = table.FRACTION
df = fulldata['11NECKLO00THFOXA'].dropna(axis=1)
SB_table['NECK'] = df.loc[:,SB_table.index].max()
df = fulldata['11CHSTLEUPTHDSXB'].dropna(axis=1)
SB_table['CHEST'] = df.loc[:,SB_table.index].min()

SB_table['COLOR'] = SB_table['CBL_BELT'].apply(lambda x: ok_color if x=='OK' else slip_color)
#%% incience - violin
plt.close('all')
plt.figure(figsize=(4,4))
plt.scatter(SB_table['FRACTION'],SB_table['CBL_BELT'],marker='.',c=SB_table['COLOR'])
violin = plt.violinplot([SB_table[SB_table.CBL_BELT.isin(['OK'])]['FRACTION'],SB_table[SB_table.CBL_BELT.isin(['SLIP'])]['FRACTION']],[0,1],vert=False,widths=[1.15,1.15],showmeans=True, bw_method=0.35)
plt.ylim(-.75,1.75)
plt.yticks((0,1),['OK','SLIP'])
plt.xlim(0.47,1)
violin['bodies'][0].set_color(ok_color)
violin['bodies'][1].set_color(slip_color)
#violin['cbars'].set_color([ok_color,slip_color])
violin['cbars'].set_color((0.15,0.15,0.15))
violin['cmeans'].set_color([ok_color,slip_color])
violin['cmins'].set_color([ok_color,slip_color])
violin['cmaxes'].set_color([ok_color,slip_color])
plt.xlabel('Belt Position Relative to Neck')
#%% incidence - hist
plt.close('all')
fig, axs = style.subplots(2,1,sharex='all', figsize=(4,4))
ok_heights, bin_edges = np.histogram(SB_table[SB_table.CBL_BELT.isin(['OK'])]['FRACTION'], bins=np.linspace(0.5,1,20), density=True)
axs[1].bar(bin_edges[:-1], ok_heights, width=np.diff(bin_edges)[0], color=ok_color, label='OK', alpha=0.5)
slip_heights, bin_edges = np.histogram(SB_table[SB_table.CBL_BELT.isin(['SLIP'])]['FRACTION'], bins=np.linspace(0.5,1,20), density=True)
axs[0].bar(bin_edges[:-1], -slip_heights, width=np.diff(bin_edges)[0], color=slip_color, label='SLIP', alpha=0.5)

plt.xlim(0.47,1)
plt.xlabel('Belt Position Relative to Neck')

axs[0].text(-0.025, 0.5, 'SLIP', horizontalalignment='right', verticalalignment='center', transform=axs[0].transAxes)
axs[1].text(-0.025, 0.5, 'OK', horizontalalignment='right', verticalalignment='center', transform=axs[1].transAxes)

axs[0].tick_params(left='off', labelleft='off', bottom='off', labelbottom='off')
axs[1].tick_params(left='off', labelleft='off')
axs[0].spines['bottom'].set_visible(False)
axs[1].spines['top'].set_visible(False)

plt.subplots_adjust(hspace=0)
#%% Neck Force
plt.close('all')
plt.figure()
plt.scatter(SB_table['FRACTION'],SB_table['NECK'],marker='.',c=SB_table['COLOR'])
#plt.figure()
#plt.scatter(SB_table['FRACTION'],SB_table['CHEST'],marker='.',c=SB_table['COLOR'])