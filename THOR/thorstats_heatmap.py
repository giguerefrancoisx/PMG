# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:08:41 2018

@author: tangk
"""
import seaborn
#%% plot heatmaps to visualize multidimensional data
full_thd_channels = []
for f in files1:
    newch = list(full_data_1[f].filter(like='11').keys())
    for n in newch:
        if not(n in full_thd_channels):
            full_thd_channels.append(n)
for f in files2:
    newch = list(full_data_2[f].filter(like='11').keys())
    for n in newch:
        if not(n in full_thd_channels):
            full_thd_channels.append(n)
            
thd1 = {}
thd2 = {}

for ch in full_thd_channels:
    a0 = pd.DataFrame(full_data_1[files1[0]],columns=[ch])
    if not((a0==0).any().iloc[0]):
        thd1[ch] = pd.DataFrame()
        thd2[ch] = pd.DataFrame()
        for f in files1:
            a = pd.DataFrame(full_data_1[f],columns=[ch]).iloc[cutoff,:].T.rename(index={ch: f})
            thd1[ch] = thd1[ch].append(a)
        for f in files2:
            a = pd.DataFrame(full_data_2[f],columns=[ch]).iloc[cutoff,:].T.rename(index={ch: f})
            thd2[ch] = thd2[ch].append(a)

thd_channels = list(thd1.keys())
thd_channels.sort()
#%% heatmaps of all channels
for j in range(0,144,20):
    fig, axs = plt.subplots(20,1,figsize=(15,60))
    for i, ax in enumerate(axs.flatten()):
        if not j+i>len(thd_channels)-1:
            seaborn.heatmap(thd1[thd_channels[j+i]].append(pd.DataFrame([np.nan]).rename(index={0: ' '}).append(thd2[thd_channels[j+i]])),ax=ax,xticklabels=False)
            ax.set_title(thd_channels[j+i])
    fig.savefig('THOR_heatmap_' + str(j+1) + '-' + str(j+i+1) + '.png',bbox_inches='tight')