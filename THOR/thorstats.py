# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:05:54 2018

Statistical tests

@author: tangk
"""
from statsmodels.sandbox.stats.runs import runstest_2samp
from scipy.stats import ks_2samp, anderson_ksamp
import read_data
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import cumfreq
import seaborn
import numpy as np
import scipy.signal as signal

# input
dir1 = 'P:\\AHEC\\Data\\THOR\\'

# OK group
#files1 = ['TC12-217',
#          'TC14-220',
#          'TC12-501',
#          'TC14-139',
#          'TC16-013',
#          'TC14-180',
#          'TC17-211',
#          'TC16-129',
#          'TC17-025',
#          'TC17-208']
files1 = ['TC12-217',
          'TC14-221',
          'TC14-241',
          'TC14-139',
          'TC14-175',
          'TC14-180',
          'TC17-211',
          'TC17-206',
          'TC15-024',
          'TC17-205',
          'TC17-204',
          'TC17-025',
          'TC17-017',
          'TC17-028']
files1.sort()

# slip group
#files2 = ['TC15-163',
#          'TC11-008',
#          'TC09-027',
#          'TC14-035',
#          'TC13-007',
#          'TC12-003',
#          'TC17-201',
#          'TC17-209',
#          'TC17-212',
#          'TC15-162'] 
files2 = ['TC14-174',
          'TC15-163',
          'TC11-008',
          'TC15-034',
          'TC12-003',
          'TC17-201',
          'TC17-210',
          'TC17-203',
          'TC17-209',
          'TC13-202',
          'TC16-125',
          'TC15-162',
          'TC13-035'] 
files2.sort()
cutoff = 1800
plotfigs = 1
subplot_dim = [6,5]
fig_size = (20,25)

#%% read data
full_data_1 = read_data.read_merged(dir1,files1)
full_data_2 = read_data.read_merged(dir1,files2)

# get lower neck x, lower neck y, T1 x, T6 x, T12 x, pelvis x, femur load, chest left upper x
channels = ['11CHSTLEUPTHDSXB',
      '11FEMRLE00THFOZB',
      '11NECKLO00THFOXA',
      '11NECKLO00THFOYA',
      '11PELV0000THACXA',
      '11SEBE0000B3FO0D',
      '11SPIN0100THACXC',
      '11SPIN1200THACXC',]
channels.sort()

chdata_1 = pd.DataFrame(index=files1,columns=channels)
chdata_2 = pd.DataFrame(index=files2,columns=channels)

for ch in channels:
    for f in files1:
        chdata_1.set_value(f,ch,pd.DataFrame(full_data_1[f],columns=[ch]).get_values().flatten()[:cutoff].tolist())
    for f in files2:
        chdata_2.set_value(f,ch,pd.DataFrame(full_data_2[f],columns=[ch]).get_values().flatten()[:cutoff].tolist())

t = full_data_1[files1[0]].iloc[:cutoff,0].get_values().flatten()
#%% get peaks

# define function to get location of peak:
def get_ipeak(data):
    m1 = min(data)
    m2 = max(data)
    i1 = np.argmin(data)
    i2 = np.argmax(data)
    
    if abs(m1) > abs(m2): # return local mind
        return i1
    else: # return local max
        return i2

# get value of peak 
def peakval(data):
    return max(abs(min(data)),abs(max(data)))

# peak acceleration (magnitude)
#peak1 = pd.DataFrame(index=files1,columns=channels)
#peak2 = pd.DataFrame(index=files2,columns=channels)

ipeak1 = chdata_1.applymap(get_ipeak)
ipeak2 = chdata_2.applymap(get_ipeak)

peak1 = chdata_1.applymap(peakval)
peak2 = chdata_2.applymap(peakval)


z_ww = {}
p_ww = {}
z_ks = {}
p_ks = {}
cdf1 = {}
cdf2 = {}

z_ad = {}
c_ad =  {}
p_ad = {}
for ch in channels:
    x1 = pd.DataFrame(peak1,columns=[ch]).get_values().flatten()
    x2 = pd.DataFrame(peak2,columns=[ch]).get_values().flatten()
    
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    
    # Wald Wolfowitz runs test
    z_ww[ch],p_ww[ch] = runstest_2samp(x1,x2)
    
    # Two-sample Kolmogorov Smirnov test
    z_ks[ch],p_ks[ch] = ks_2samp(x1,x2)
    
    z_ad[ch],c_ad[ch],p_ad[ch] = anderson_ksamp([x1,x2])

    # empirical CDF
    cdf1[ch] = cumfreq(x1)
    cdf2[ch] = cumfreq(x2)
    
#%% plot characteristics
# peak values
fig, axs = plt.subplots(2,4,figsize=(20,10))
for i, ax in enumerate(axs.flatten()):
    ax.bar(['OK','Slip'],[np.mean(peak1.iloc[:,i]),np.mean(peak2.iloc[:,i])])
    ax.plot(['OK'],[peak1.iloc[:,i]],'.')
    ax.plot(['Slip'],[peak2.iloc[:,i]],'.')
    ax.errorbar(['OK','Slip'],[np.mean(peak1.iloc[:,i]),np.mean(peak2.iloc[:,i])],yerr=[np.std(peak1.iloc[:,i]),np.std(peak2.iloc[:,i])],ecolor='black',capsize=5,linestyle='none')
    ax.set_title(channels[i])

# empirical CDFs
fig, axs = plt.subplots(2,4,sharey='all',figsize=(20,10))
for i, ax in enumerate(axs.flatten()):
    c1 = cdf1[list(cdf1.keys())[i]]
    c2 = cdf2[list(cdf2.keys())[i]]
    ax.plot(c1.lowerlimit+np.linspace(0,c1.binsize*c1.cumcount.size,c1.cumcount.size),c1.cumcount,label='Belts OK')
    ax.plot(c2.lowerlimit+np.linspace(0,c2.binsize*c2.cumcount.size,c2.cumcount.size),c2.cumcount,label='Belts Slip')
    ax.legend()
    ax.set_title(channels[i])

#%% plot peaks
if plotfigs:
    for ch in channels:
        fig, axs = plt.subplots(subplot_dim[0],subplot_dim[1],sharey='all',figsize=fig_size)
        fig.suptitle(ch)
        raw = pd.DataFrame(chdata_1,columns=[ch]).append(pd.DataFrame([np.nan])).append(pd.DataFrame(chdata_2,columns=[ch]))
        ipeak = pd.DataFrame(ipeak1,columns=[ch]).append(pd.DataFrame([np.nan])).append(pd.DataFrame(ipeak2,columns=[ch]))
        
        for j, ax in enumerate(axs.flatten()):
            if j > len(raw)-1:
                break
            if np.isnan(ipeak.iloc[j,0]):
                continue
            ax.plot(t,raw.iloc[j,0],label='Raw Data')
            ax.plot(t[int(ipeak.iloc[j,0])],raw.iloc[j,0][int(ipeak.iloc[j,0])],'.r',markersize=10,label='Peak')
            ax.set_title(raw.index[j])
            ax.legend()
#%% get times to peak
channels_to_plot = ['11HEAD0000THACXA',
                    '11SPIN0100THACXC',
                    '11CHST0000THACXC',
                    '11SPIN1200THACXC',
                    '11PELV0000THACXA',
                    '11FEMRLE00THFOZB',
                    '11FEMRRI00THFOZB']   
chdata_1 = pd.DataFrame(index=files1,columns=channels_to_plot)
chdata_2 = pd.DataFrame(index=files2,columns=channels_to_plot) 

for ch in channels_to_plot:
    for f in files1:
        chdata_1.set_value(f,ch,pd.DataFrame(full_data_1[f],columns=[ch]).get_values().flatten()[:cutoff].tolist())
    for f in files2:
        chdata_2.set_value(f,ch,pd.DataFrame(full_data_2[f],columns=[ch]).get_values().flatten()[:cutoff].tolist())

def smooth_data(data):
    if data==[]:
        return []
    if abs(min(data)) > abs(max(data)):
        return signal.savgol_filter(np.negative(data),301, 5)
    else:
        return signal.savgol_filter(data,201,5)
def get_i2peak(data):
    if data.size==0:
        return []
    peaks_indices = signal.find_peaks_cwt(data,[100])
    if len(peaks_indices)>1:
        peaks_vals = data[peaks_indices]
        peaks_indices = peaks_indices[peaks_vals>max(peaks_vals)-5]
        peaks_vals = peaks_vals[peaks_vals>max(peaks_vals)-5]
    return int(peaks_indices[0]) 
    
smth1 = chdata_1.applymap(smooth_data)
smth2 = chdata_2.applymap(smooth_data)

i2peak1 = smth1.applymap(get_i2peak)
i2peak2 = smth2.applymap(get_i2peak)


if plotfigs:
    for ch in channels_to_plot:
        fig, axs = plt.subplots(subplot_dim[0],subplot_dim[1],sharey='all',figsize=fig_size)
        fig.suptitle(ch)
        raw = pd.DataFrame(chdata_1,columns=[ch]).append(pd.DataFrame(chdata_2,columns=[ch]))
        smth = pd.DataFrame(smth1,columns=[ch]).append(pd.DataFrame(smth2,columns=[ch]))
        i2peak = pd.DataFrame(i2peak1,columns=[ch]).append(pd.DataFrame(i2peak2,columns=[ch]))
        
        for j, ax in enumerate(axs.flatten()):
            ax.plot(t,raw.iloc[j,0],label='Raw Data')
            ax.plot(t,np.negative(smth.iloc[j,0]).T, label='Smoothed')
            ax.plot(t[i2peak.iloc[j,0]],raw.iloc[j,0][i2peak.iloc[j,0]],'.r',markersize=10,label='Time to Peak')
            ax.set_title(raw.index[j])
#        fig.savefig('Peak_Detection_' + ch + '.png',bbox_inches='tight')




#%% stats on times to peak
z_ww = {}
p_ww = {}
z_ks = {}
p_ks = {}
cdf1 = {}
cdf2 = {}
t2peak1 = pd.DataFrame(index=files1)
t2peak2 = pd.DataFrame(index=files2)
for ch in channels_to_plot:
    t1 = t[pd.DataFrame(i2peak1,columns=[ch]).get_values().flatten()] 
    t2 = t[pd.DataFrame(i2peak2,columns=[ch]).get_values().flatten()] 
    
    t2peak1.insert(0,ch,t1)
    t2peak2.insert(0,ch,t2)

    # Wald Wolfowitz runs test
    z_ww[ch],p_ww[ch] = runstest_2samp(t1,t2)
    
    # Two-sample Kolmogorov Smirnov test
    z_ks[ch],p_ks[ch] = ks_2samp(t1,t2)

    # empirical CDF
    cdf1[ch] = cumfreq(t1)
    cdf2[ch] = cumfreq(t2)

#%% plot characteristics
# times to peak
fig, axs = plt.subplots(2,3,figsize=(15,10))
for i, ax in enumerate(axs.flatten()):
    if i > len(channels_to_plot)-1:
        break
    ax.bar(['OK','Slip'],[np.mean(t2peak1.iloc[:,i]),np.mean(t2peak2.iloc[:,i])])
    ax.errorbar(['OK','Slip'],[np.mean(t2peak1.iloc[:,i]),np.mean(t2peak2.iloc[:,i])],yerr=[np.std(t2peak1.iloc[:,i]),np.std(t2peak2.iloc[:,i])],ecolor='black',capsize=5,linestyle='none')
    ax.set_title(t2peak1.keys()[i])

# empirical CDFs
fig, axs = plt.subplots(2,3,sharey='all',figsize=(15,10))
for i, ax in enumerate(axs.flatten()):
    if i > len(channels_to_plot)-1:
        break
    c1 = cdf1[list(cdf1.keys())[i]]
    c2 = cdf2[list(cdf2.keys())[i]]
    ax.plot(c1.lowerlimit+np.linspace(0,c1.binsize*c1.cumcount.size,c1.cumcount.size),c1.cumcount,label='Belts OK')
    ax.plot(c2.lowerlimit+np.linspace(0,c2.binsize*c2.cumcount.size,c2.cumcount.size),c2.cumcount,label='Belts Slip')
    ax.legend()
    ax.set_title(list(cdf1.keys())[i])
#%% plot heatmaps to visualize multidimensional data
full_thd_channels = list(full_data_1[files1[0]].filter(like='11').keys())
    
thd1 = {}
thd2 = {}

for ch in full_thd_channels:
    a0 = pd.DataFrame(full_data_1[files1[0]],columns=[ch])
    if not((a0==0).any().iloc[0]):
        thd1[ch] = pd.DataFrame()
        thd2[ch] = pd.DataFrame()
        for f in files1:
            a = pd.DataFrame(full_data_1[f],columns=[ch]).iloc[:cutoff,:].T.rename(index={ch: f})
            thd1[ch] = thd1[ch].append(a)
        for f in files2:
            a = pd.DataFrame(full_data_2[f],columns=[ch]).iloc[:cutoff,:].T.rename(index={ch: f})
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

#%% heatmaps of selected acceleration channels in x--compare global responses across samples

fig, axs = plt.subplots(nrows=10,ncols=2,figsize=(40,20))
#for i, ax in enumerate(axs.flatten()):
#    if i<=9:
#        seaborn.heatmap(pd.DataFrame(full_data_1[files1[i]],index=range(1600),columns=channels_to_plot).T,ax=ax,xticklabels=False,yticklabels=False,vmin=-80,vmax=10)
#        ax.set_title(files1[i])
#    else:
#        seaborn.heatmap(pd.DataFrame(full_data_2[files2[i-10]],index=range(1600),columns=channels_to_plot).T,ax=ax,xticklabels=False,yticklabels=False,vmin=-80,vmax=10)
#        ax.set_title(files2[i-10])
for i, ax in enumerate(axs):
    seaborn.heatmap(pd.DataFrame(full_data_1[files1[i]],index=range(1600),columns=channels_to_plot).T,ax=ax[0],xticklabels=False,yticklabels=False,vmin=-80,vmax=10,cmap='Spectral')
    ax[0].set_title(files1[i])
    seaborn.heatmap(pd.DataFrame(full_data_2[files2[i]],index=range(1600),columns=channels_to_plot).T,ax=ax[1],xticklabels=False,yticklabels=False,vmin=-80,vmax=10,cmap='Spectral')
    ax[1].set_title(files2[i])


        