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
import PMG.COM.data as data
from PMG.COM.data import check_and_clean
import PMG.COM.plotstyle as style
import PMG.COM.table as tb

THOR = os.fspath('P:/AHEC/Data/THOR/')
chlist = ['11CHSTLEUPTHDSXB', '11CHSTRIUPTHDSXB', '11CHSTRILOTHDSXB', '11CHSTLELOTHDSXB']

time, fulldata = openHDF5(THOR, chlist)

for ch in chlist:
    print(ch)
    fulldata[ch] = check_and_clean(fulldata[ch],1)

table = tb.get('THOR')
table = table[table.TYPE.isin(['Frontale/Véhicule']) & table.VITESSE.isin([48,56])]
ok, slip = tb.tcns(tb.split(table, column='CBL_BELT', categories=['OK','SLIP']))

#%%
if 0:
    cd = style.colordict(chlist)
    titles = dict(zip(chlist, ['Upper Left', 'Upper Right', 'Lower Right', 'Lower Left']))

    plt.close('all')
    r, c = 2, 2
    fig, axs = style.subplots(r, c, sharex='all', sharey='all')
    ok_df = pd.DataFrame()
    slip_df = pd.DataFrame()
    for i, ch in enumerate(chlist):
        ax=axs[i]
        peaks, times = data.find_peak(data.smooth(fulldata[ch]), time)
        ok_df = pd.concat([ok_df, peaks.loc[ok].dropna().abs()], axis=1)
        slip_df = pd.concat([slip_df, peaks.loc[slip].dropna().abs()], axis=1)

        hist, bin_edges = np.histogram(np.array(peaks.loc[ok].dropna().abs()), bins=np.linspace(0,60,41), normed=True)
        ax.bar(bin_edges[1:], hist, 60/40, label=titles[ch]+' ok', color=cd[ch], alpha=0.25)
        hist, bin_edges = np.histogram(np.array(peaks.loc[slip].dropna().abs()), bins=np.linspace(0,60,41), normed=True)
        ax.bar(bin_edges[1:], -hist, 60/40, label=titles[ch]+' slip', color=cd[ch], alpha=0.75)

        ax.set_title(titles[ch])
        ax.set_xlabel('Displacement')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid()

    ok_df.columns = chlist
    slip_df.columns = chlist
    ok_sum = ok_df.sum(axis=1)
    slip_sum = slip_df.sum(axis=1)
    plt.figure()
    ax=plt.gca()

    hist, bin_edges = np.histogram(np.array(ok_sum), bins=np.linspace(40,120,21), normed=True)
    ax.bar(bin_edges[1:], hist, 80/20, label='ok', color=cd[ch], alpha=0.25)
    hist, bin_edges = np.histogram(np.array(slip_sum), bins=np.linspace(40,120,21), normed=True)
    ax.bar(bin_edges[1:], -hist, 80/20, label='slip', color=cd[ch], alpha=0.75)
    ax.legend()
    ax.grid()

    fig = plt.figure()
    ax=plt.gca()
    for i, ch in enumerate(chlist):
        peaks, times = data.find_peak(data.smooth(fulldata[ch]), time)

        hist, bin_edges = np.histogram(np.array(peaks.loc[ok].dropna().abs()), bins=np.linspace(0,60,41), normed=True)
        plt.bar(bin_edges[1:], hist, 60/40, label=titles[ch]+' ok', color=cd[ch], alpha=0.25)
        hist, bin_edges = np.histogram(np.array(peaks.loc[slip].dropna().abs()), bins=np.linspace(0,60,41), normed=True)
        plt.bar(bin_edges[1:], -hist, 60/40, label=titles[ch]+' slip', color=cd[ch], alpha=0.75)
    ax.set_title('Histogram of Displacements')
    ax.set_xlabel('Displacement')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.grid()


#%%
if 0:
    categories=['Upper Left', 'Upper Right', 'Lower Right', 'Lower Left']
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ticks = np.linspace(10,60,6).astype(np.int)
    #tick_labels = list(map(str,ticks))

    fig, axs = style.subplots(1,2, subplot_kw=dict(projection='polar'))

    for ax in axs:
        ax.set_theta_offset(np.pi*3/4)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_rlabel_position(135)

    axs[0].set_title('Ok')
    axs[1].set_title('Slip')
    #plt.ylim(0,40)
    for ax, df in zip(axs, [ok_df, slip_df]):
        for tcn in df.index:
            values = df.loc[tcn].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=tcn)
            ax.set_yticks(ticks)
    style.maximize()

#%%
if 1:
    plt.close('all')
    colors = ['tab:blue', 'tab:orange']
    s = 2
    ticks = np.linspace(0,s,7).round(1)#.astype(np.int)
    tick_labels = list(map(str,ticks))

    fig, axs = style.subplots(1,2, subplot_kw=dict(projection='polar'), figsize=(7.4,3))
    plt.suptitle('Chest deflections')
    for ax in axs:
        ax.set_theta_offset(np.pi*3/4)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.xaxis.set_label_coords(1.5,0.5) #TODO change this for polar plots
        ax.set_rlabel_position(135)
        ax.set_yticks(ticks)
#        ax.set_yticklabels(tick_labels, color='grey')
        ax.tick_params(labelleft='off')
#        ax.text((135+4)/180*np.pi, 1.04*s, 'Fraction of mean')

        ax.plot(np.linspace(0, 2*np.pi,180), s/2*np.ones(180,), color='k', linewidth=0.5)

    #scaler = pd.concat([ok_df,slip_df]).max(axis=0)/s
    scaler = ok_df.mean(axis=0)/(s/2)

    ok_square = np.array((ok_df/scaler).mean())
    slip_square = np.array((slip_df/scaler).mean())
    ok_square = np.append(ok_square, ok_square[0])
    slip_square = np.append(slip_square, slip_square[0])

    ok_std = np.array((ok_df/scaler).std())/2
    slip_std = np.array((slip_df/scaler).std())/2
    ok_std = np.append(ok_std, ok_std[0])
    slip_std = np.append(slip_std, slip_std[0])

    ax = axs[0]
    for i, df in enumerate([ok_df, slip_df]):

        for tcn in df.index:
            df2 = df/scaler#df.max(axis=0)
            values = df2.loc[tcn].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', color=colors[i], alpha=0.5)

    ax = axs[1]
    ax.plot(angles, ok_square, color=colors[0], linewidth=2)
    ax.plot(angles, slip_square, color=colors[1], linewidth=2)

    for ax in axs:
        ax.plot([np.nan], [np.nan], label='No-Slip', color=colors[0])
        ax.plot([np.nan], [np.nan], label='Slip', color=colors[1])
        ax.set_ylim(0,1.5)#s)
    axs[-1].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.subplots_adjust(top=0.8,bottom=0.1,left=0.0,right=0.86,hspace=0,wspace=0)
