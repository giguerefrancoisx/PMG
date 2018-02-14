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
from PMG.COM.outliers import check_and_clean
import PMG.COM.plotstyle as style
import PMG.COM.table as tb

THOR = os.fspath('P:/AHEC/Data/THOR/')
chlist = ['11CHSTLEUPTHDSXB', '11CHSTRIUPTHDSXB', '11CHSTRILOTHDSXB', '11CHSTLELOTHDSXB']

time, fulldata = openHDF5(THOR, chlist)

for ch in chlist:
    print(ch)
    fulldata[ch] = check_and_clean(fulldata[ch])

ok, slip = tb.tcns(tb.split(tb.get('THOR'), column='CBL_BELT', categories=['OK','SLIP']))

#%%
cd = style.colordict(chlist)
titles = dict(zip(chlist, ['Upper Left', 'Upper Right', 'Lower Right', 'Lower Left']))

plt.close('all')
r, c = 2, 2
fig, axs = style.subplots(r, c, sharex='all', sharey='all')
ok_df = pd.DataFrame()
slip_df = pd.DataFrame()
for i, ch in enumerate(chlist):
    ax=axs[i]
    peaks, times = data.find_peaks(data.smooth(fulldata[ch]), time)
    ok_df = pd.concat([ok_df, peaks.loc[ok].dropna().abs()], axis=1)
    slip_df = pd.concat([slip_df, peaks.loc[slip].dropna().abs()], axis=1)
#    peaks.loc[ok].hist(ax=ax, normed=True, label='ok', color=cd[ch], alpha=0.25, bins=np.linspace(-60,20,81))
#    peaks.loc[slip].hist(ax=ax, normed=True, label='slip', color=cd[ch], alpha=0.75, bins=np.linspace(-60,20,81))

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
#ok_sum.hist(ax=ax, normed=True, label='ok', alpha=0.25, bins=np.linspace(-120,-40,21))
#slip_sum.hist(ax=ax, normed=True, label='slip', alpha=0.75, bins=np.linspace(-120,-40,21))
hist, bin_edges = np.histogram(np.array(ok_sum), bins=np.linspace(40,120,21), normed=True)
ax.bar(bin_edges[1:], hist, 80/20, label='ok', color=cd[ch], alpha=0.25)
hist, bin_edges = np.histogram(np.array(slip_sum), bins=np.linspace(40,120,21), normed=True)
ax.bar(bin_edges[1:], -hist, 80/20, label='slip', color=cd[ch], alpha=0.75)
ax.legend()
ax.grid()

fig = plt.figure()
ax=plt.gca()
for i, ch in enumerate(chlist):
    peaks, times = data.find_peaks(data.smooth(fulldata[ch]), time)
#    peaks.loc[ok].hist(ax=ax, normed=True, label=titles[ch]+' ok', color=cd[ch], alpha=0.25, bins=np.linspace(-60,20,81))
#    peaks.loc[slip].hist(ax=ax, normed=True, label=titles[ch]+' slip', color=cd[ch], alpha=0.75, bins=np.linspace(-60,20,81)).invert_yaxis()
    hist, bin_edges = np.histogram(np.array(peaks.loc[ok].dropna().abs()), bins=np.linspace(0,60,41), normed=True)
    plt.bar(bin_edges[1:], hist, 60/40, label=titles[ch]+' ok', color=cd[ch], alpha=0.25)
    hist, bin_edges = np.histogram(np.array(peaks.loc[slip].dropna().abs()), bins=np.linspace(0,60,41), normed=True)
    plt.bar(bin_edges[1:], -hist, 60/40, label=titles[ch]+' slip', color=cd[ch], alpha=0.75)
ax.set_title('Histogram of Displacements')
ax.set_xlabel('Displacement')
ax.set_ylabel('Frequency')
ax.legend()
plt.grid()

#style.maximize()

#%%
# Set data
#df = pd.DataFrame({
#'var1': [38, 1.5, 30, 4],
#'var2': [29, 10, 9, 34],
#'var3': [8, 39, 23, 24],
#'var4': [7, 31, 33, 14],
#'var5': [28, 15, 32, 14]
#})
#plt.close('all')
# number of variable
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
#plt.close('all')
colors = ['tab:blue', 'tab:orange']
s = 2
ticks = np.linspace(0,s,7).round(1)#.astype(np.int)
tick_labels = list(map(str,ticks))

fig, axs = style.subplots(1,2, subplot_kw=dict(projection='polar'))
for ax in axs:
    ax.set_theta_offset(np.pi*3/4)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_rlabel_position(135)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, color='grey')
    ax.set_title('Chest deflections')
    ax.text((135+2)/180*np.pi, 1.02*s, 'Fraction of mean')

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
#ax.fill_between(angles, ok_square+ok_std, ok_square-ok_std, color=colors[0], linewidth=2, alpha=0.25)
#ax.fill_between(angles, slip_square+slip_std, slip_square-slip_std, color=colors[1], linewidth=2, alpha=0.25)

for ax in axs:
    ax.plot([np.nan], [np.nan], label='Ok', color=colors[0])
    ax.plot([np.nan], [np.nan], label='Slip', color=colors[1])
    ax.set_ylim(0,s)
    ax.legend()
