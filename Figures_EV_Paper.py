# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:40:26 2019

@author: giguerf
"""
import numpy as np
import pandas as pd
#import matplotlib.colors as clr
import matplotlib.pyplot as plt
from PMG.read_data import initialize
import PMG.COM.plotstyle as style

chlist = ['11HEAD0000THACXA', '11HEAD0000THACRA', '11NECKUP00THFOXA', '11CHST0000THACXC', '11CHST0000THACRC', '11PELV0000THACXA',
          '11HEAD0000H3ACXA', '11HEAD0000H3ACRA', '11NECKUP00H3FOXA', '11CHST0000H3ACXC', '11CHST0000H3ACRC', '11PELV0000H3ACXA',
          'Max_11HEAD0000THACRA','Max_11HEAD0000H3ACRA','Min_11HEAD0000THACXA','Min_11HEAD0000H3ACXA',
          'Max_11NECKUP00THFOZA','Max_11NECKUP00H3FOZA','Max_11CHST0000THACRC','Max_11CHST0000H3ACRC',
          'Min_11CHST0000THACXC','Min_11CHST0000H3ACXC','Min_11PELV0000THACXA','Min_11PELV0000H3ACXA',
          '10CVEHCG0000ACXD','10SIMELE00INACXD','10SIMERI00INACXD']

chlist = list(set(chlist))

table, t, chdata = initialize('P:/Data Analysis/Projects/AHEC EV/', chlist, range(100,1600))
#%%
table.Type_2 = table.Type_2.fillna(table.Type)
pairs = table[table.Type.isin(['ICE'])&table.Speed.isin([48])&table.Series_1.isin([1])].Pair.reset_index()
pairs.columns = ['ICE','EV']

series2 = table[table.Speed.isin([48])&table.Series_2.isin([1])].iloc[:,:10].reset_index()
s_left = series2.loc[series2.Type.isin(['ICE']),['TC','Model','Counterpart']]
s_right = series2.loc[:,['TC','Model','Counterpart']]
pairs2 = s_left.merge(s_right,left_on='Model', right_on='Counterpart').loc[:,['TC_x','TC_y']]
pairs2.columns = ['ICE','EV']

plt.rcParams['font.size'] = 10

H3_color  = '#fdc086'#[0.6843137254901961, 0.6176470588235294, 0.8]#'#5e3c99'
TH_color  = '#beaed4'#[0.996078431372549, 0.8607843137254902, 0.6941176470588235]#'#fdb863'
TH2_color = '#7fc97f'#[0.9509803921568627, 0.6901960784313725, 0.5019607843137255]#'#e66101'

ice_color = '#ca0020'
ev_color  = '#0571b0'

def make_rgb_transparent(rgb, bg_rgb, alpha):
    return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, bg_rgb)]
import matplotlib.colors as clr
dots = make_rgb_transparent(clr.to_rgba('#386cb0'), clr.to_rgba('white'), 0.75)

#cmap_neck = clr.LinearSegmentedColormap.from_list('custom', [ok_color,slip_color,slip_color], 256)
#colors_neck = style.colordict(fulldata['11NECKLO00THFOXA'].loc[:,slips+oks], 'max', cmap_neck)
#cmap_chst = clr.LinearSegmentedColormap.from_list('custom', [ok_color,ok_color,slip_color,slip_color], 256)
#colors_chst = style.colordict(fulldata['11CHSTLEUPTHDSXB'].loc[:,slips+oks], 'min', cmap_chst)
#%% Data Setup for all plots

ch_pairs = [['Max_11HEAD0000THACRA','Max_11HEAD0000H3ACRA'],
                 ['Min_11HEAD0000THACXA','Min_11HEAD0000H3ACXA'],
                 ['Max_11NECKUP00THFOZA','Max_11NECKUP00H3FOZA'],
                 ['Max_11CHST0000THACRC','Max_11CHST0000H3ACRC'],
                 ['Min_11CHST0000THACXC','Min_11CHST0000H3ACXC'],
                 ['Min_11PELV0000THACXA','Min_11PELV0000H3ACXA']]

th_chs = ['Max_11HEAD0000THACRA','Min_11HEAD0000THACXA',
          'Max_11NECKUP00THFOZA','Max_11CHST0000THACRC',
          'Min_11CHST0000THACXC','Min_11PELV0000THACXA']

h3_chs = ['Max_11HEAD0000H3ACRA','Min_11HEAD0000H3ACXA',
          'Max_11NECKUP00H3FOZA','Max_11CHST0000H3ACRC',
          'Min_11CHST0000H3ACXC','Min_11PELV0000H3ACXA']

combined_chs = ['Head X', 'Head R', 'Neck X', 'Chest X', 'Chest R', 'Pelv X']

chlist = ['Max_11HEAD0000THACRA','Max_11HEAD0000H3ACRA','Min_11HEAD0000THACXA','Min_11HEAD0000H3ACXA',
          'Max_11NECKUP00THFOZA','Max_11NECKUP00H3FOZA','Max_11CHST0000THACRC','Max_11CHST0000H3ACRC',
          'Min_11CHST0000THACXC','Min_11CHST0000H3ACXC','Min_11PELV0000THACXA','Min_11PELV0000H3ACXA']

features = pd.read_csv('P:/Data Analysis/Projects/AHEC EV/features.csv', index_col=0)
# Pair up data
merged = pairs.merge(features.reset_index(), left_on='ICE', right_on='index').loc[:,['ICE','EV','index']+chlist]
merged2 = merged.merge(features.reset_index(), left_on='EV', right_on='index', suffixes=('ICE','EV'))
pair_data = merged2.loc[:,['ICE','EV']+[ch+'ICE' for ch in chlist]+[ch+'EV' for ch in chlist]]

# Pair up data (Series 2)
merged = pairs2.merge(features.reset_index(), left_on='ICE', right_on='index').loc[:,['ICE','EV','index']+th_chs]
merged2 = merged.merge(features.reset_index(), left_on='EV', right_on='index', suffixes=('ICE','EV'))
pair_data_2 = merged2.loc[:,['ICE','EV']+[ch+'ICE' for ch in th_chs]+[ch+'EV' for ch in th_chs]]

# Combine H3 and TH channels and pair up
#to_combine = features.loc[:,chlist]
#TH_features = to_combine.loc[:,th_chs].rename(columns=dict(zip(th_chs,combined_chs)))
#H3_features = to_combine.loc[:,h3_chs].rename(columns=dict(zip(h3_chs,combined_chs)))
#combined_features = TH_features.combine_first(H3_features)
#merged = pairs.merge(combined_features.reset_index(), left_on='ICE', right_on='index').loc[:,['ICE','EV','index']+combined_chs]
#merged2 = merged.merge(combined_features.reset_index(), left_on='EV', right_on='index', suffixes=('ICE','EV'))
#combine_pair_data = merged2.loc[:,['ICE','EV']+[ch+'ICE' for ch in combined_chs]+[ch+'EV' for ch in combined_chs]]


#%% Setup for Figure 1

#ch_groups = [['11HEAD0000THACXA', '11HEAD0000THACRA', '11NECKUP00THFOXA', '11CHST0000THACXC', '11CHST0000THACRC', '11PELV0000THACXA'],
#             ['11HEAD0000H3ACXA', '11HEAD0000H3ACRA', '11NECKUP00H3FOXA', '11CHST0000H3ACXC', '11CHST0000H3ACRC', '11PELV0000H3ACXA']]

chname = ['Head X', 'Head R', 'Neck X', 'Chest X', 'Chest R', 'Pelv X']
chlist = ['Max_11HEAD0000THACRA','Max_11HEAD0000H3ACRA','Min_11HEAD0000THACXA','Min_11HEAD0000H3ACXA',
          'Max_11NECKUP00THFOZA','Max_11NECKUP00H3FOZA','Max_11CHST0000THACRC','Max_11CHST0000H3ACRC',
          'Min_11CHST0000THACXC','Min_11CHST0000H3ACXC','Min_11PELV0000THACXA','Min_11PELV0000H3ACXA']

data = {}
for ch in chlist:
    data[ch] = pair_data.loc[:,ch+'ICE']-pair_data.loc[:,ch+'EV']

data2 = {}
for ch in th_chs:
    data2[ch] = pair_data_2.loc[:,ch+'ICE']-pair_data_2.loc[:,ch+'EV']

#%% Figure 1 - Boxplot with scatter/jitter
H3_color  = '#1b9e77'
TH_color  = '#d95f02'
TH2_color = '#7570b3'

plt.close('all')
fig, axs = style.subplots(2, 3, sharex='col', sharey='all', figsize=(6.5,5), visible=False)
axs = list(axs)
ax2 = axs[2].twinx() #secondary axis for neck force

for i, (h3, th, name) in enumerate(zip(h3_chs, th_chs, chname)):
    if i == 2:
        ax = ax2
    else:
        ax = axs[i]

#    H3_array = np.arange(-5,1,1)
#    TH_array = np.arange(5,11,1)
#    TH2_array = np.arange(-3,8,1)
    H3_array = data[h3].dropna().values
    TH_array = data[th].dropna().values
    TH2_array = data2[th].dropna().values

    boxplot_stacked_data = [H3_array,TH_array,TH2_array]

    for j, (y, c) in enumerate(zip(boxplot_stacked_data,[H3_color,TH_color,TH2_color])):
        x = np.random.normal(j+1, 0.025, len(y))
        ax.plot(x, y, '.', color=dots, zorder=4, markeredgecolor='k', markeredgewidth=.5)

    props = ax.boxplot(boxplot_stacked_data, vert=True, whis=1.5, widths=0.75, patch_artist=True,
                       showfliers=False, showmeans=True, meanline=True,
                       meanprops=dict(linestyle='-', linewidth=1, color='red'))

    for patch, color in zip(props['boxes'], [H3_color, TH_color, TH2_color]):
        patch.set_facecolor(color)#, alpha=0.5)
        patch.set_alpha(0.5)
    for line in props['medians']:
        line.set_color((0.25,0.25,0.25))

for i, (ax, name) in enumerate(zip(axs, chname)):
    ax.text(0.5, 1.05, name, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
    ax.patch.set_alpha(0)
    ax.axhline(color=(0.45,0.45,0.45), linewidth=1, zorder=1)
#    ax.spines['top'].set_visible(False)
    ax.tick_params(left=False)
#    ax.set_xticklabels(['H3', 'TH', 'TH2'])
    ax.set_xticklabels(['H3', 'THOR', 'THOR\nS2'], fontsize=8)
    ax.set_xlim((0,4))

    if i in [1,4,5]:
        ax.spines['left'].set_edgecolor((0.85,0.85,0.85))
        ax.tick_params(left=False)

    if i in [0,3]:
        ax.tick_params(left=True)
        ax.set_ylabel('Difference in\nAcceleration [g]')

    if i in [0,1,2]:
        ax.tick_params(labelbottom=True)

    if i in [2]:
#        ax2.set_ylim((-20,20))
        ax2.tick_params(right=True, labelright=True)
        ax2.set_ylabel('Difference in Force [N]')

plt.subplots_adjust(top=0.920,bottom=0.110,left=0.14,right=0.86, hspace=0.45, wspace=0.0)
#%% Setup for Figure 2

chlist = ['10CVEHCG0000ACXD','10SIMELE00INACXD']#,'10SIMERI00INACXD']
data = {}
for ch in chlist:
    data[ch] = {}
    for tc in table.index.tolist():#pairs.ICE.tolist()+pairs.EV.tolist():#+pairs2.ICE.tolist()+pairs2.EV.tolist():
        data[ch][tc] = chdata[ch].get(tc)
    data[ch] = pd.DataFrame(data[ch])

fig2_table = table[table.Speed.isin([48])&table.Series_1.isin([1])]
groups = fig2_table.groupby('Type_2').groups
groups['EV'].tolist()

#%% Figure 2 - Ice vs EV quantile plots VECG and SIME
#ice_color = '#ca0020'
#ev_color  = '#0571b0'
al=0.3


plt.close('all')
labels = ['Centre of Gravity ACx',
          'Left Side Member ACx']#,'Right Side Member $\mathregular{A_x}$ [g]']
ylabel = dict(zip(chlist, labels))

fig, axs = style.subplots(1, 2, sharex='all', sharey='all', figsize=(7,3))

for i, ch in enumerate(chlist):
    df_ice = data[ch].get(pairs.ICE.tolist())
    df_ev = data[ch].get(pairs.EV.tolist())

    window = 50
    alpha = 0.10
    ice_median = df_ice.median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ev_median = df_ev.median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ice_high = df_ice.quantile(1-alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ice_low = df_ice.quantile(alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ev_high = df_ev.quantile(1-alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ev_low = df_ev.quantile(alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()


    axs[i].plot(t, ice_median, color=ice_color, label='ICE Median, n={}'.format(len(pairs.ICE.tolist())))
    axs[i].plot(t, ev_median, color=ev_color, label='EV Median, n={}'.format(len(pairs.EV.tolist())))
    axs[i].fill_between(t, ice_high, ice_low, color=ice_color, alpha=al, label='ICE {:2.0f}$^{{{}}}$ Percentile'.format(100*(1-alpha),'th'))
    axs[i].fill_between(t, ev_high, ev_low, color=ev_color, alpha=al, label='EV {:2.0f}$^{{{}}}$ Percentile'.format(100*(1-alpha),'th'))

    axs[i].set_title(ylabel[ch])
    axs[i].set_ylabel('Acceleration [g]')
#    axs[i].text(0.04,0.94,'ABC'[i], horizontalalignment='left',
#       verticalalignment='top',transform=axs[i].transAxes, fontsize=12)

axs[0].set_ylim(-50,20)
axs[0].set_yticks(np.linspace(-50,20,8))

axs[0].set_xlim(0,0.15)
axs[0].set_xticks(np.linspace(0,0.15,6))
axs[0].set_xlabel('Time [s]')
axs[1].set_xlabel('Time [s]')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, 'upper center', ncol=2, fontsize=9,
           bbox_to_anchor = (0.5, 1), bbox_transform = fig.transFigure)

plt.subplots_adjust(top=0.705,bottom=0.16,left=0.1,right=0.97,hspace=0.45,wspace=0.35)
#%% Figure 3 - Ice vs EV/HYB/PHEV quantile plots VECG and SIME
#ice_color = '#ca0020'
#ev_color  = '#0571b0'
ice_color = 'red'
al=0.3


plt.close('all')
labels = ['Centre of Gravity ACx',
          'Left Side Member ACx']#,'Right Side Member $\mathregular{A_x}$ [g]']
ylabel = dict(zip(chlist, labels))

fig, axs = style.subplots(3, 2, sharex='all', sharey='all', figsize=(7,8))
axs = np.array(axs).reshape(3,2)

for i, ch in enumerate(chlist):
    df_ice = data[ch].get(groups['ICE'].tolist())

    ice_median = df_ice.median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ice_high = df_ice.quantile(1-alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ice_low = df_ice.quantile(alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()

    window = 50
    alpha = 0.10
    for j, (color, label, tcns) in enumerate([['blue', 'EV', groups['EV'].tolist()],
                                              ['purple','HYB', groups['HYBRID'].tolist()],
                                              ['magenta','PHEV', groups['PHEV'].tolist()]]):
        df = data[ch].get(tcns)
        median = df.median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
        high = df.quantile(1-alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
        low = df.quantile(alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()

        axs[j,i].plot(t, ice_median, color=ice_color, label='ICE Median, n={}'.format(len(groups['ICE'].tolist())))
        axs[j,i].fill_between(t, ice_high, ice_low, color=ice_color, alpha=al, label='ICE {:2.0f}$^{{{}}}$ Percentile'.format(100*(1-alpha),'th'))

        axs[j,i].plot(t, median, color=color, label='{} Median, n={}'.format(label, len(tcns)))
        axs[j,i].fill_between(t, high, low, color=color, alpha=al, label='{} {:2.0f}$^{{{}}}$ Percentile'.format(label,100*(1-alpha),'th'))

        axs[j,0].set_ylabel('Acceleration [g]')
#        if not j==2:
#            axs[-1,i].fill_between(t, np.nan, np.nan, color=color, alpha=al, label='{} {:2.0f}$^{{{}}}$ Percentile'.format(label,100*(1-alpha),'th'))
    axs[0,i].set_title(ylabel[ch])

#    axs[i].text(0.04,0.94,'ABC'[i], horizontalalignment='left',
#       verticalalignment='top',transform=axs[i].transAxes, fontsize=12)

axs[0,0].set_ylim(-50,20)
axs[0,0].set_yticks(np.linspace(-50,20,8))

axs[-1,1].set_xlim(0,0.15)
axs[-1,1].set_xticks(np.linspace(0,0.15,6))
axs[-1,1].set_xlabel('Time [s]')
axs[-1,0].set_xlabel('Time [s]')

handles1, labels1 = axs[0,0].get_legend_handles_labels()
handles2, labels2 = axs[1,0].get_legend_handles_labels()
handles3, labels3 = axs[2,0].get_legend_handles_labels()
handles, labels = handles1+handles2[1::2]+handles3[1::2], labels1+labels2[1::2]+labels3[1::2]
fig.legend(handles, labels, 'upper center', ncol=2, fontsize=9,
           bbox_to_anchor = (0.5, 1), bbox_transform = fig.transFigure)

plt.subplots_adjust(top=0.840,bottom=0.070,left=0.1,right=0.97,hspace=0.315,wspace=0.35)
#%% Figure 4 - Ice vs EV/HYB/PHEV quantile plots VECG and SIME (By lines)
#ice_color = '#ca0020'
#ev_color  = '#0571b0'
ice_color = 'red'
al=0.3


plt.close('all')
labels = ['Centre of Gravity ACx',
          'Left Side Member ACx']#,'Right Side Member $\mathregular{A_x}$ [g]']
ylabel = dict(zip(chlist, labels))

fig, axs = style.subplots(3, 2, sharex='all', sharey='all', figsize=(7,8))
axs = np.array(axs).reshape(3,2)

for i, ch in enumerate(chlist):
    df_ice = data[ch].get(groups['ICE'].tolist())

    ice_median = df_ice.median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ice_high = df_ice.quantile(1-alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
    ice_low = df_ice.quantile(alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()

    window = 50
    alpha = 0.10
    for j, (color, label, tcns) in enumerate([['blue', 'EV', groups['EV'].tolist()],
                                              ['purple','HYB', groups['HYBRID'].tolist()],
                                              ['magenta','PHEV', groups['PHEV'].tolist()]]):
        df = data[ch].get(tcns)
        median = df.median(axis=1).rolling(window,0,center=True,win_type='triang').mean()
        high = df.quantile(1-alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()
        low = df.quantile(alpha/2, axis=1).rolling(window,0,center=True,win_type='triang').mean()

        axs[j,i].plot(t, ice_median, color=ice_color, label='ICE Median, n={}'.format(len(groups['ICE'].tolist())))
#        axs[j,i].plot(t, df_ice, color=ice_color, label='ICE, n={}'.format(len(groups['ICE'].tolist())))
        axs[j,i].fill_between(t, ice_high, ice_low, color=ice_color, alpha=al, label='ICE {:2.0f}$^{{{}}}$ Percentile'.format(100*(1-alpha),'th'))

        axs[j,i].plot(t, df, color=color, alpha=0.6, label='{}, n={}'.format(label, len(tcns)))
#        axs[j,i].fill_between(t, high, low, color=color, alpha=al, label='{} {:2.0f}$^{{{}}}$ Percentile'.format(label,100*(1-alpha),'th'))

        axs[j,0].set_ylabel('Acceleration [g]')
#        if not j==2:
#            axs[-1,i].fill_between(t, np.nan, np.nan, color=color, alpha=al, label='{} {:2.0f}$^{{{}}}$ Percentile'.format(label,100*(1-alpha),'th'))
    axs[0,i].set_title(ylabel[ch])

#    axs[i].text(0.04,0.94,'ABC'[i], horizontalalignment='left',
#       verticalalignment='top',transform=axs[i].transAxes, fontsize=12)

axs[0,0].set_ylim(-50,20)
axs[0,0].set_yticks(np.linspace(-50,20,8))

axs[-1,1].set_xlim(0,0.15)
axs[-1,1].set_xticks(np.linspace(0,0.15,6))
axs[-1,1].set_xlabel('Time [s]')
axs[-1,0].set_xlabel('Time [s]')

#handles1, labels1 = axs[0,0].get_legend_handles_labels()
#handles2, labels2 = axs[1,0].get_legend_handles_labels()
#handles3, labels3 = axs[2,0].get_legend_handles_labels()
#handles, labels = handles1+handles2[1::2]+handles3[1::2], labels1+labels2[1::2]+labels3[1::2]
#fig.legend(handles, labels, 'upper center', ncol=2, fontsize=9,
#           bbox_to_anchor = (0.5, 1), bbox_transform = fig.transFigure)

plt.subplots_adjust(top=0.840,bottom=0.070,left=0.1,right=0.97,hspace=0.315,wspace=0.35)

