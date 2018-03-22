# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:51:04 2018

@author: giguerf
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import PMG.COM.data as dat
import PMG.COM.table as tb
import PMG.COM.plotstyle as style
THOR = 'P:/AHEC/Data/THOR/'
chlist = []
chlist.extend(['11NECKLO00THFOXA','11NECKLO00THFOYA','11CHSTLEUPTHDSXB','11FEMRLE00THFOZB','11CHSTRILOTHDSXB'])
chlist.extend(['11HEAD0000THACXA','11SPIN0100THACXC', '11CHST0000THACXC', '11SPIN1200THACXC', '11PELV0000THACXA'])
chlist.extend(['11HEAD0000THACYA','11SPIN0100THACYC', '11CHST0000THACYC', '11SPIN1200THACYC', '11PELV0000THACYA'])
chlist.extend(['11SPIN0100THACYC','11THSP0100THAVXA','11THSP0100THAVZA'])
time, fulldata = dat.import_data(THOR, chlist, check=False)
table = tb.get('THOR')
table = table[table.TYPE.isin(['Frontale/Véhicule'])]
slips  = table[table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
oks = table[table.CBL_BELT.isin(['OK'])].CIBLE.tolist()

plt.rcParams['font.size']= 10

slip_color = 'tab:blue'
ok_color = 'tab:red'

cmap_neck = matplotlib.colors.LinearSegmentedColormap.from_list('custom', [ok_color,slip_color,slip_color], 256)
colors_neck = style.colordict(fulldata['11NECKLO00THFOXA'].loc[:,slips+oks], 'max', cmap_neck)
cmap_chst = matplotlib.colors.LinearSegmentedColormap.from_list('custom', [ok_color,ok_color,slip_color,slip_color], 256)
colors_chst = style.colordict(fulldata['11CHSTLEUPTHDSXB'].loc[:,slips+oks], 'min', cmap_chst)

#%% FIGURE - DIFFERENCE IN RESPONSE BETWEEN SLIP/OK
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

        axs[0+2*i].set_ylabel(ylabel[channel])
        axs[0+2*i].yaxis.set_label_coords(-0.28,0.5)
        axs[0+2*i].yaxis.set_major_formatter(xfmt)

    axs[-1].set_xlim(0,0.3)
    axs[-1].set_xlabel('Time [s]')
    axs[-2].set_xlabel('Time [s]')

    style.legend(axs[-2], loc='lower right', fontsize=6)
    axs[-1].legend(loc='lower right', fontsize=6)

    plt.tight_layout()
#%% Like above but box plot
#if 1:
#    plt.close('all')
#    chlist = ['11NECKLO00THFOXA','11NECKLO00THFOYA','11CHSTLEUPTHDSXB','11FEMRLE00THFOZB']
#    labels = ['Lower Neck $\mathregular{F_x}$ [N]',
#              'Lower Neck $\mathregular{F_y}$ [N]',
#              'Upper Left Chest $\mathregular{D_x}$ [mm]',
#              'Left Femur $\mathregular{F_x}$ [N]']
#    ylabel = dict(zip(chlist, labels))
#    xfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
#    xfmt.set_powerlimits((-3,4))
#
#    n = len(chlist)
#
#    fig, axs = style.subplots(n, 1, sharex='all', sharey='row', figsize=(6.5,2*n))
#
#    for i, channel in enumerate(chlist):
#        df = fulldata[channel].dropna(axis=1)
#        df = df.loc[:,slips+oks]
#        for tcn in slips:
#            axs[0+2*i].plot(time, df.loc[:,tcn], color=slip_color, lw=1, label='Slip')
#        for tcn in oks:
#            axs[0+2*i].plot(time, df.loc[:,tcn], color=ok_color, lw=1, label='No-Slip')
#
#        window = 100
#        alpha = 0.10
#        slip_median = df.loc[:,slips].median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
#        ok_median = df.loc[:,oks].median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
#        slip_high = df.loc[:,slips].quantile(1-alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
#        slip_low = df.loc[:,slips].quantile(alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
#        ok_high = df.loc[:,oks].quantile(1-alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
#        ok_low = df.loc[:,oks].quantile(alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
#
#
#        axs[1+2*i].plot(time, slip_median, color=slip_color, label='Median, n={}'.format(len(slips)))
#        axs[1+2*i].plot(time, ok_median, color=ok_color, label='Median, n={}'.format(len(oks)))
#        axs[1+2*i].fill_between(time, slip_high, slip_low, color=slip_color, alpha=0.2, label='{:2.0f}th Percentile'.format(100*(1-alpha)))
#        axs[1+2*i].fill_between(time, ok_high, ok_low, color=ok_color, alpha=0.2, label='{:2.0f}th Percentile'.format(100*(1-alpha)))
#    #    axs[1+2*i].legend(loc='lower right', fontsize=6)
#
#        axs[0+2*i].set_ylabel(ylabel[channel])
#        axs[0+2*i].yaxis.set_label_coords(-0.28,0.5)
#        axs[0+2*i].yaxis.set_major_formatter(xfmt)
#
#    axs[-1].set_xlim(0,0.3)
#    axs[-1].set_xlabel('Time [s]')
#    axs[-2].set_xlabel('Time [s]')
#
#    style.legend(axs[-2], loc='lower right', fontsize=6)
#    axs[-1].legend(loc='lower right', fontsize=6)
#
#    plt.tight_layout()
#%% FIGURE - BOOTSTRAPPED TIME TO PEAK - HISTOGRAM
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

    allpeaks = {}
    alltimes = {}
    for ch in chlist:
        for group, label in zip([slips, oks], ['Slip', 'No-Slip']):
            df = fulldata[ch].dropna(axis=1).loc[:,group].dropna(axis=1)
            sm = df#dat.smooth_peaks(df)
            allpeaks[ch+label], alltimes[ch+label] = dat.find_peak(sm, time)

    colors = dict(zip(['Slip', 'No-Slip'], [slip_color, ok_color]))

    plt.close('all')
    fig, axs = style.subplots(len(chlist), 1, sharex='all', sharey='all', figsize=(5,5), visible=False)

    for i, (ch, name) in enumerate(zip(chlist, chname)):

        for group, label in zip([slips, oks], ['Slip', 'No-Slip']):
            df = fulldata[ch].dropna(axis=1).loc[:,group].dropna(axis=1)
            peaks, times = allpeaks[ch+label], alltimes[ch+label]
            x1_r = bootstrap_resample(times.values, n=5000)
            means = x1_r.mean(axis=1)
#            heights, bin_edges = np.histogram(means, bins=np.linspace(0.065, 0.105, 80), density=True)
#            axs[i].bar(bin_edges[:-1], heights, width=np.diff(bin_edges)[0], color=colors[label], label=label, alpha=0.5)
            style.hist(axs[i], means, np.linspace(0.065, 0.105, 80), color=colors[label], label=label, alpha=0.5)

    axs[-1].set_ylim(10,400)
    axs[-1].set_xlim(0.065,0.105)
    axs[-1].set_xlabel('Times [s]')
    axs[-1].legend(loc='lower right')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

### Convert the above to Ridgeline Plot
    overlap = 0.50
    for ax, name in zip(axs, chname):
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
#%% FIGURE - BOOTSTRAPPED TIME TO PEAK - BOXPLOT
if 0:
    chlist = ['11HEAD0000THACXA','11SPIN0100THACXC', '11CHST0000THACXC', '11SPIN1200THACXC', '11PELV0000THACXA']
#    chlist = ['11HEAD0000THACYA','11SPIN0100THACYC', '11CHST0000THACYC', '11SPIN1200THACYC', '11PELV0000THACYA']
    chname = ['Head', 'Spine T1', 'Chest', 'Spine T12', 'Pelvis']

    allpeaks = {}
    alltimes = {}
    for ch in chlist:
        for group, label in zip([slips, oks], ['Slip', 'No-Slip']):
            df = fulldata[ch].dropna(axis=1).loc[:,group].dropna(axis=1)
            sm = df#dat.smooth_peaks(df)
            allpeaks[ch+label], alltimes[ch+label] = dat.find_peak(sm, time)

    colors = dict(zip(['Slip', 'No-Slip'], [slip_color, ok_color]))

    plt.close('all')
    fig, axs = style.subplots(len(chlist), 1, sharex='all', sharey='all', figsize=(5,5), visible=False)

    for i, (ch, name) in enumerate(zip(chlist, chname)):

        df = fulldata[ch].dropna(axis=1).loc[:,slips].dropna(axis=1)
        peaks, times = allpeaks[ch+'Slip'], alltimes[ch+'Slip']
        x1_r = bootstrap_resample(times.values, n=5000)
        means_slip = x1_r.mean(axis=1)

        df = fulldata[ch].dropna(axis=1).loc[:,oks].dropna(axis=1)
        peaks, times = allpeaks[ch+'No-Slip'], alltimes[ch+'No-Slip']
        x1_r = bootstrap_resample(times.values, n=5000)
        means_ok = x1_r.mean(axis=1)

        means = np.vstack((means_ok,means_slip)).T
        props = axs[i].boxplot(means, vert=False, whis='range', widths=0.75, patch_artist=True)
        for patch, color in zip(props['boxes'], [ok_color, slip_color]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for line in props['medians']:
            line.set_color((0.25,0.25,0.25))

    for color, label in zip([slip_color, ok_color], ['Slip', 'No-Slip']):
        axs[-1].fill_between(np.array([np.nan]),np.array([np.nan]),np.array([np.nan]), color=color, label=label, alpha=0.5)
    axs[-1].legend(loc='lower right')

    axs[-1].set_ylim(0.5,2.5)
    axs[-1].set_xlim(0.060,0.11)
#    axs[-1].set_xlim(0.05,0.115)#axs[-1].set_xlim(0.045,0.16) #without head/with head
    axs[-1].set_xlabel('Time [s]')
    plt.tight_layout()

    for ax, name in zip(axs, chname):
        ax.text(-0.01, 0.5, name, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        ax.patch.set_alpha(0)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_edgecolor((0.85,0.85,0.85))
        ax.tick_params(left='off', labelleft='off', bottom='off')

    axs[0].spines['top'].set_visible(True)
    axs[-1].tick_params(bottom='on')
    axs[-1].spines['bottom'].set_edgecolor((0,0,0))

    plt.subplots_adjust(left=0.177, top=0.97, hspace=0)
#%% FIGURE - BOOTSTRAPPED PEAK - BOXPLOT
if 0:
    chlist = ['11HEAD0000THACXA','11SPIN0100THACXC', '11CHST0000THACXC', '11SPIN1200THACXC', '11PELV0000THACXA']
#    chlist = ['11HEAD0000THACYA','11SPIN0100THACYC', '11CHST0000THACYC', '11SPIN1200THACYC', '11PELV0000THACYA']
    chname = ['Head', 'Spine T1', 'Chest', 'Spine T12', 'Pelvis']

    allpeaks = {}
    alltimes = {}
    for ch in chlist:
        for group, label in zip([slips, oks], ['Slip', 'No-Slip']):
            df = fulldata[ch].dropna(axis=1).loc[:,group].dropna(axis=1)
            sm = df#dat.smooth_peaks(df)
            allpeaks[ch+label], alltimes[ch+label] = dat.find_peak(sm, time)

    colors = dict(zip(['Slip', 'No-Slip'], [slip_color, ok_color]))

    plt.close('all')
    fig, axs = style.subplots(len(chlist), 1, sharex='all', sharey='all', figsize=(5,5), visible=False)

    for i, (ch, name) in enumerate(zip(chlist, chname)):

        df = fulldata[ch].dropna(axis=1).loc[:,slips].dropna(axis=1)
        peaks, times = allpeaks[ch+'Slip'], alltimes[ch+'Slip']
        x1_r = bootstrap_resample(peaks.values, n=5000)
        means_slip = x1_r.mean(axis=1)

        df = fulldata[ch].dropna(axis=1).loc[:,oks].dropna(axis=1)
        peaks, times = allpeaks[ch+'No-Slip'], alltimes[ch+'No-Slip']
        x1_r = bootstrap_resample(peaks.values, n=5000)
        means_ok = x1_r.mean(axis=1)

        means = np.vstack((means_ok,means_slip)).T
        props = axs[i].boxplot(means, vert=False, whis='range', widths=0.75, patch_artist=True)
        for patch, color in zip(props['boxes'], [ok_color, slip_color]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for line in props['medians']:
            line.set_color((0.25,0.25,0.25))

    for color, label in zip([slip_color, ok_color], ['Slip', 'No-Slip']):
        axs[-1].fill_between(np.array([np.nan]),np.array([np.nan]),np.array([np.nan]), color=color, label=label, alpha=0.5)
    axs[-1].legend(loc='lower left')

    axs[-1].set_ylim(0.5,2.5)
    axs[-1].set_xlim(-85,-25)
#    axs[-1].set_xlim(7.5,27.5)
    axs[-1].set_xlabel('Acceleration [g]')
    plt.tight_layout()

    for ax, name in zip(axs, chname):
        ax.text(-0.01, 0.5, name, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        ax.patch.set_alpha(0)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_edgecolor((0.85,0.85,0.85))
        ax.tick_params(left='off', labelleft='off', bottom='off')

    axs[0].spines['top'].set_visible(True)
    axs[-1].tick_params(bottom='on')
    axs[-1].spines['bottom'].set_edgecolor((0,0,0))

    plt.subplots_adjust(left=0.177, top=0.97, hspace=0)
#%% FIGURES - SEAT BELT POSITION FRACTION
columns = ['CIBLE','CBL_BELT','T1','T2','FRACTION']
SB_table = table.loc[table.FRACTION.dropna().index,columns].set_index('CIBLE')

SB_table['NECK'] = fulldata['11NECKLO00THFOXA'].loc[:,SB_table.index].max()
SB_table['CHEST'] = fulldata['11CHSTLEUPTHDSXB'].loc[:,SB_table.index].min()

SB_table['COLOR'] = SB_table['CBL_BELT'].apply(lambda x: ok_color if x=='OK' else slip_color)
#%% fraction vs incidence - violin
plt.close('all')
plt.figure(figsize=(4,4))
plt.scatter(SB_table['FRACTION'],SB_table['CBL_BELT'],marker='.',c=SB_table['COLOR'])
violin = plt.violinplot([SB_table[SB_table.CBL_BELT.isin(['OK'])]['FRACTION'],SB_table[SB_table.CBL_BELT.isin(['SLIP'])]['FRACTION']],[0,1],vert=False,widths=[1.15,1.15],showmeans=True, bw_method=0.35)
plt.ylim(-.75,1.75)
plt.yticks((0,1),['OK','SLIP'])
plt.xlim(0.47,1)
violin['bodies'][0].set_color(ok_color)
violin['bodies'][1].set_color(slip_color)
violin['cbars'].set_color((0.15,0.15,0.15))
violin['cmeans'].set_color([ok_color,slip_color])
violin['cmins'].set_color([ok_color,slip_color])
violin['cmaxes'].set_color([ok_color,slip_color])
plt.xlabel('Belt Position Relative to Neck')
#%% fraction vs incidence - hist
plt.close('all')
fig, axs = style.subplots(2,1,sharex='all', figsize=(4,4))
style.hist(axs[1], SB_table[SB_table.CBL_BELT.isin(['OK'])]['FRACTION'], bins=np.linspace(0.5,1,20), color=ok_color, label='OK', alpha=0.5)
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
#%% fraction vs incidence - hist 2
#TODO get more data and increase n_bins
plt.close('all')
plt.figure(figsize=(4,4))
ax = plt.gca()
style.hist(ax, SB_table[SB_table.CBL_BELT.isin(['OK'])]['FRACTION'], bins=np.linspace(0.5,1,10), color=ok_color, label='OK', alpha=0.5)
style.hist(ax, SB_table[SB_table.CBL_BELT.isin(['SLIP'])]['FRACTION'], bins=np.linspace(0.5,1,10), color=slip_color, label='SLIP', alpha=0.5)

ax.tick_params(left='off', labelleft='off')
ax.legend(loc='upper right')
plt.xlim(0.47,1)
plt.xlabel('Belt Position Relative to Neck')
plt.ylabel('Frequency')

#plt.close('all')
plt.figure(figsize=(4,4))
ax = plt.gca()
ok_heights, bin_edges = np.histogram(SB_table[SB_table.CBL_BELT.isin(['OK'])]['FRACTION'], bins=np.linspace(0.5,1,7))
slip_heights, bin_edges = np.histogram(SB_table[SB_table.CBL_BELT.isin(['SLIP'])]['FRACTION'], bins=np.linspace(0.5,1,7))
ax.bar(bin_edges[:-1], ok_heights/(slip_heights+ok_heights+0.00001), width=np.diff(bin_edges)[0], color=ok_color, label='OK', alpha=0.5)
ax.bar(bin_edges[:-1], slip_heights/(slip_heights+ok_heights+0.00001), width=np.diff(bin_edges)[0], color=slip_color, label='SLIP', alpha=0.5)

ax.tick_params(left='off', labelleft='off')
ax.legend(loc='upper right')
plt.xlim(0.47,1)
plt.xlabel('Belt Position Relative to Neck')
plt.ylim(0,1.25)
plt.ylabel('Relative Frequency')
#%% Neck Force
plt.close('all')
plt.figure()
plt.scatter(SB_table['FRACTION'],SB_table['NECK'],marker='.',c=SB_table['COLOR'])
plt.figure()
plt.scatter(SB_table['FRACTION'],SB_table['CHEST'],marker='.',c=SB_table['COLOR'])
#%% Time to Slip
plt.close('all')
plt.figure()
notnull = ~SB_table['T1'].isin(['?'])
plt.scatter(SB_table['FRACTION'][notnull],SB_table['T1'][notnull],marker='.',c=SB_table['COLOR'][notnull])
#%% FIGURE - TIME TO BELT SLIP VS NECK/CHEST RESPONSE
if 0:
    ok, slip = tb.split(table, 'CBL_BELT', ['OK','SLIP']).values()
    slip = slip[~slip.T1.isnull() & ~slip.T1.isin(['?'])]
    bin1 = slip[(0.060<=slip.T1) & (slip.T1<0.075)]

    plt.close('all')
    fig, axs = style.subplots(2, 2, sharex='all', sharey='col', figsize=(6.5, 6.5))

    ind = pd.concat([time, pd.Series(time.index)], axis=1).set_index('Time')
    chlist = ['11NECKLO00THFOXA','11CHSTLEUPTHDSXB']
    chname = dict(zip(chlist, ['Lower Neck $F_x$', 'Upper Left Chest $D_x$']))

    for c, (ch,group) in enumerate(zip(['11NECKLO00THFOXA', '11CHSTLEUPTHDSXB','11NECKLO00THFOXA', '11CHSTLEUPTHDSXB'],[bin1,bin1,ok,ok])):

        ax=axs[c]
        tcns = group.CIBLE.tolist()
        df = fulldata[ch].loc[:,tcns]
        if c in [0,2]:
            colors = colors_neck
        if c in [1,3]:
            colors = colors_chst

        for tcn in df.columns:
            ax.plot(time, df[tcn], color=colors[tcn])

        if c in [0,1]:
            tab = group.set_index('CIBLE')
            points = tab.loc[tcns].T1
            points2 = tab.loc[tcns].T2
            values = np.diagonal(df.loc[(points*10000+100).values.astype(int),points.index])
            values2 = np.diagonal(df.loc[(points2*10000+100).values.astype(int),points2.index])

            ax.plot(points, values, '.', color='b')
            ax.plot(points2, values2, '.', color='r')
            axs[c].set_title(chname[ch])
        if c in [2,3]:
            axs[c].set_xlabel('Time [s]')

    axs[0].set_ylabel('Early Slip')
    axs[2].set_ylabel('No-Slip Belts')
    axs[-1].set_xlim(0,0.2)
#%% FIGURE - TIME TO BELT SLIP VS NECK/CHEST RESPONSE - 3 ROWS
if 0:
    ok, slip = tb.split(table, 'CBL_BELT', ['OK','SLIP']).values()
    slip = slip[~slip.T1.isnull() & ~slip.T1.isin(['?'])]
    bin1 = slip[(0.060<=slip.T1) & (slip.T1<0.08)]
    bin2 = slip[(0.080<slip.T1) & (slip.T1<0.105)]

    plt.close('all')
    fig, axs = style.subplots(3, 2, sharex='all', sharey='col', figsize=(6.5, 7))

    ind = pd.concat([time, pd.Series(time.index)], axis=1).set_index('Time')
    chlist = ['11NECKLO00THFOXA','11CHSTLEUPTHDSXB']
    chname = dict(zip(chlist, ['Lower Neck $F_x$', 'Upper Left Chest $D_x$']))

    for c, (ch,group) in enumerate(zip(['11NECKLO00THFOXA', '11CHSTLEUPTHDSXB','11NECKLO00THFOXA', '11CHSTLEUPTHDSXB','11NECKLO00THFOXA', '11CHSTLEUPTHDSXB'],[bin1,bin1,bin2,bin2,ok,ok])):

        ax=axs[c]
        tcns = group.CIBLE.tolist()
        df = fulldata[ch].loc[:,tcns]
        if c in [0,2,4]:
            colors = colors_neck
        if c in [1,3,5]:
            colors = colors_chst

        for tcn in df.columns:
            ax.plot(time, df[tcn], color=colors[tcn])

        if c in [0,1,2,3]:
            tab = group.set_index('CIBLE')
            points = tab.loc[tcns].T1
            points2 = tab.loc[tcns].T2
            values = np.diagonal(df.loc[(points*10000+100).values.astype(int),points.index])
            values2 = np.diagonal(df.loc[(points2*10000+100).values.astype(int),points2.index])

            ax.plot(points, values, '.', color='b')
            ax.plot(points2, values2, '.', color='r')
        if c in [0,1]:
            axs[c].set_title(chname[ch])
        if c in [4,5]:
            axs[c].set_xlabel('Time [s]')

    axs[0].set_ylabel('Early Slip Belts')
    axs[2].set_ylabel('Late Slip Belts')
    axs[4].set_ylabel('No-Slip Belts')
    axs[-1].set_xlim(0,0.2)
    plt.subplots_adjust(top=0.947,bottom=0.083,left=0.14,right=0.953,hspace=0.28,wspace=0.302)
#%% TABLE NECK FORCE PEAK LOAD
table = tb.get('THOR')
speeds = [[48,56],[48]]#,[56]]
print('X')
for speed in speeds:
    slips = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
    oks = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['OK'])].CIBLE.tolist()
    print(round(fulldata['11NECKLO00THFOXA'].loc[:,oks].max().mean(), -1),'+',round(fulldata['11NECKLO00THFOXA'].loc[:,oks].max().std(), -1))
    print(round(fulldata['11NECKLO00THFOXA'].loc[:,slips].max().mean(), -1),'+',round(fulldata['11NECKLO00THFOXA'].loc[:,slips].max().std(), -1))
    print(round(fulldata['11NECKLO00THFOXA'].loc[:,slips].max().mean()-fulldata['11NECKLO00THFOXA'].loc[:,oks].max().mean(), -1))
print('Y')
for speed in speeds:
    slips = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
    oks = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['OK'])].CIBLE.tolist()
    print(round(fulldata['11NECKLO00THFOYA'].loc[:,oks].min().mean(), -1),'+',round(fulldata['11NECKLO00THFOYA'].loc[:,oks].min().std(), -1))
    print(round(fulldata['11NECKLO00THFOYA'].loc[:,slips].min().mean(), -1),'+',round(fulldata['11NECKLO00THFOYA'].loc[:,slips].min().std(), -1))
    print(round(fulldata['11NECKLO00THFOYA'].loc[:,slips].min().mean()-fulldata['11NECKLO00THFOYA'].loc[:,oks].min().mean(), -1))
#%% TABLE CHEST DSX PEAK
speeds = [[48,56],[48]]#,[56]]

print('LEUP mean')
for speed in speeds:
    slips = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
    oks = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['OK'])].CIBLE.tolist()
    print(round(fulldata['11CHSTLEUPTHDSXB'].loc[:,oks].min().mean(), 0),'+',round(fulldata['11CHSTLEUPTHDSXB'].loc[:,oks].min().std(), 0))
    print(round(fulldata['11CHSTLEUPTHDSXB'].loc[:,slips].min().mean(), 0),'+',round(fulldata['11CHSTLEUPTHDSXB'].loc[:,slips].min().std(), 0))
    print(round(fulldata['11CHSTLEUPTHDSXB'].loc[:,slips].min().mean()-fulldata['11CHSTLEUPTHDSXB'].loc[:,oks].min().mean(), 0))
print('LEUP median')
for speed in speeds:
    slips = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
    oks = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['OK'])].CIBLE.tolist()
    print(round(fulldata['11CHSTLEUPTHDSXB'].loc[:,oks].min().median(), 0))
    print(round(fulldata['11CHSTLEUPTHDSXB'].loc[:,slips].min().median(), 0))
    print(round(fulldata['11CHSTLEUPTHDSXB'].loc[:,slips].min().median()-fulldata['11CHSTLEUPTHDSXB'].loc[:,oks].min().median(), 0))
print('RILO mean')
for speed in speeds:
    slips = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
    oks = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['OK'])].CIBLE.tolist()
    print(round(fulldata['11CHSTRILOTHDSXB'].loc[:,oks].min().mean(), 0),'+',round(fulldata['11CHSTRILOTHDSXB'].loc[:,oks].min().std(), 0))
    print(round(fulldata['11CHSTRILOTHDSXB'].loc[:,slips].min().mean(), 0),'+',round(fulldata['11CHSTRILOTHDSXB'].loc[:,slips].min().std(), 0))
    print(round(fulldata['11CHSTRILOTHDSXB'].loc[:,slips].min().mean()-fulldata['11CHSTRILOTHDSXB'].loc[:,oks].min().mean(), 0))
print('RILO median')
for speed in speeds:
    slips = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
    oks = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin(speed) & table.CBL_BELT.isin(['OK'])].CIBLE.tolist()
    print(round(fulldata['11CHSTRILOTHDSXB'].loc[:,oks].min().median(), 0))
    print(round(fulldata['11CHSTRILOTHDSXB'].loc[:,slips].min().median(), 0))
    print(round(fulldata['11CHSTRILOTHDSXB'].loc[:,slips].min().median()-fulldata['11CHSTRILOTHDSXB'].loc[:,oks].min().median(), 0))
#%% TABLE OTHER CHANNELS

#Right clavicle stuff

df = fulldata['11SPIN0100THACYC'].loc[:,oks].max()
print(round(df.mean(),0), round(df.std(),0))
df = fulldata['11SPIN0100THACYC'].loc[:,slips].max()
print(round(df.mean(),0), round(df.std(),0))

df = fulldata['11CHST0000THACYC'].loc[:,oks].max()
print(round(df.mean(),0), round(df.std(),0))
df = fulldata['11CHST0000THACYC'].loc[:,slips].max()
print(round(df.mean(),0), round(df.std(),0))

df = fulldata['11THSP0100THAVXA'].loc[:,oks].min()
print(round(df.mean(),0), round(df.std(),0))
df = fulldata['11THSP0100THAVXA'].loc[:,slips].min()
print(round(df.mean(),0), round(df.std(),0))

df = fulldata['11THSP0100THAVZA'].loc[:,oks].max()
print(round(df.mean(),0), round(df.std(),0))
df = fulldata['11THSP0100THAVZA'].loc[:,slips].max()
print(round(df.mean(),0), round(df.std(),0))

df = fulldata['11FEMRLE00THFOZB'].loc[:,oks].min()
print(round(df.mean(),0), round(df.std(),0))
df = fulldata['11FEMRLE00THFOZB'].loc[:,slips].min()
print(round(df.mean(),0), round(df.std(),0))
