# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:05:54 2018

Statistical tests

@author: tangk
"""
from scipy.stats import anderson_ksamp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import cumfreq
import numpy as np
import PMG.COM.arrange as arrange
import PMG.COM.plotfuns as plotfuns
import PMG.COM.bootstrap as bootstrap
from scipy.signal import medfilt 



#%% get anderson-darling statistic
for p in ['peak','t2peak']:
    for ch in channels:
        for i in ['-tive','+tive']: # min and max
            x1 = arrange.get_values(props1[p][ch][i][:-1].get_values().astype(float))
            x2 = arrange.get_values(props2[p][ch][i][:-1].get_values().astype(float))
    
            # two sample Anderson-Darling test
            props1['stats'].set_value(p,(ch,i),anderson_ksamp([x1,x2]))
            props2['stats'].set_value(p,(ch,i),anderson_ksamp([x1,x2]))
    
            # empirical CDF
            props1[p].set_value('cdf',(ch,i),cumfreq(x1))
            props2[p].set_value('cdf',(ch,i),cumfreq(x2))  

#%% plot characteristics
# peak values
if plotfigs:
    for p in ['peak','t2peak']:
        for i,ch in enumerate(channels):
            fig = plt.figure(figsize=(5,5))
            ax = plt.axes()
            x1 = np.abs(arrange.get_values(props1[p][ch][wherepeaks[i]][:-1].get_values().astype(float)))
            x2 = np.abs(arrange.get_values(props2[p][ch][wherepeaks[i]][:-1].get_values().astype(float)))
            ax = plotfuns.plot_bar(ax,['OK','Slip'],x1,x2)
            ax.set_title(p + '-' + ch)
            if savefigs:
                fig.savefig(writename + p + '_' + ch + '_' + wherepeaks[i] + '_bar.png', bbox_inches='tight')
            plt.show()
            plt.close(fig)
        
            # ecdfs
            fig = plt.figure(figsize=(5,5))
            ax = plt.axes()
            ax = plotfuns.plot_ecdf(ax,['OK','Slip'],
                                    props1[p][ch][wherepeaks[i]]['cdf'],
                                    props2[p][ch][wherepeaks[i]]['cdf'])
            ax.set_title(p + '-' + ch)
            if savefigs:
                fig.savefig(writename + p + '_' + ch + '_' + wherepeaks[i] + '_ecdf.png', bbox_inches='tight')
            plt.show()
            plt.close(fig)

            # plot peaks
            if len(files1)%10==0:
                nnan = 0
            else:
                nnan = 10-len(files1)%10
            if p=='peak':
                ip = 'ipeak'
                peaklabel = 'Peak'
            elif p=='t2peak':
                ip = 'i2peak'
                peaklabel = 'Time to Peak'
            
            raw = chdata_1[ch].append(pd.Series(np.tile([np.nan],nnan))).append(chdata_2[ch])
            if p=='t2peak': # also get smoothed value
                smth = props1['smth'][ch].append(pd.Series(np.tile([np.nan],nnan))).append(props2['smth'][ch])
            ipeak = props1[ip][ch][wherepeaks[i]].append(pd.Series(np.tile([np.nan],nnan))).append(props2[ip][ch][wherepeaks[i]])

            n1 = int(np.ceil(len(files1)/10))
            n2 = int(np.ceil(len(files2)/10))
            fig, axs = plt.subplots(n1+n2,10,sharey='all',figsize=(40,4*(n1+n2)))        
            for j, ax in enumerate(axs.flatten()[range(len(raw))]):
                ax.set_title(raw.index[j])
                if np.isnan(raw[raw.index[j]]).all():
                    continue
                ax.plot(t,raw[raw.index[j]],label='Raw Data')
                if p=='t2peak':
                    ax.plot(t,smth[raw.index[j]],label='Smoothed Data')
                ax.plot(t[ipeak[raw.index[j]]],np.array(raw[raw.index[j]])[ipeak[raw.index[j]]],'.r',markersize=10,label=peaklabel)
                ax.legend()
            fig.suptitle(p + '-' + ch)
            plt.show()
            if savefigs:
                fig.savefig(writename + p + '_' + ch + '_' + wherepeaks[i] + '.png',bbox_inches='tight')
            plt.close(fig)        
#%% try bootstrap
for p in ['peak','t2peak']:
    for i,ch in enumerate(channels):
        X1 = arrange.get_values(props1[p][ch][wherepeaks[i]][:-1].get_values().astype(float))
        X1_resample = bootstrap.bootstrap_resample(X1,n=bs_n_it)
        
        # centered bootstrap percentile confidence interval
        L1, U1 = bootstrap.get_ci(X1,X1_resample,bs_alpha)
        b1 = bootstrap.get_bins_from_ci(np.mean(X1_resample,axis=1),[L1,U1],bs_nbin[0])
    
        X2 = arrange.get_values(props2[p][ch][wherepeaks[i]][:-1].get_values().astype(float))
        X2_resample = bootstrap.bootstrap_resample(X2,n=bs_n_it)
            
        L2,U2 = bootstrap.get_ci(X2,X2_resample,bs_alpha)
        b2 = bootstrap.get_bins_from_ci(np.mean(X2_resample,axis=1),[L2,U2],bs_nbin[1])
        
        BSp = bootstrap.test(X1,X2,nbs=bs_n_it)
        props1['stats'].set_value('BSp_' + p,(ch,wherepeaks[i]),BSp)
        props2['stats'].set_value('BSp_' + p,(ch,wherepeaks[i]),BSp)
        
        fig = plt.figure()
        ax = plt.axes()
        ax = plotfuns.plot_bs_distribution(ax,(np.mean(X1_resample,axis=1),b1,'OK'),(np.mean(X2_resample,axis=1),b2,'Slip'),[[L1,U1],[L2,U2]])
        ax.set_title(p + '-' + ch)
        plt.show()
        
        if savefigs:
            fig.savefig(writename + 'BSMeanDistribution_' + p + '_' + ch + '_' + wherepeaks[i] + '.png',bbox_inches='tight')
        plt.close(fig)
        
        fig = plt.figure(figsize=(3,3))
        ax = plt.axes()
        mean1 = np.mean(np.mean(X1_resample,axis=1))
        mean2 = np.mean(np.mean(X2_resample,axis=1))
        ax.bar(['OK','Slip'],[mean1,mean2],0.7,color=[0.4, 0.4, 0.4])
        ax.errorbar(['OK','Slip'],[mean1,mean2],[[mean1-L1,mean2-L2],[U1-mean1,U2-mean2]],ecolor='black',capsize=5,linestyle='none')
        ax.set_title(p + '-' + ch)
        plt.rc('font',size=12)
        plt.show()
        if savefigs:
            fig.savefig(writename + 'BSBar_' + p + '_' + ch + '_' + wherepeaks[i] + '.png',bbox_inches='tight')
        plt.close(fig)
#%%
if writefiles:
    props1['stats'].to_csv(writename + channels[0] + 'stat.csv')    

#%% compare stds
std1 = pd.DataFrame(np.concatenate(chdata_1['11NECKLO00THFOXA'].get_values()).reshape(len(chdata_1.index),-1),index=chdata_1.index)
std2 = pd.DataFrame(np.concatenate(chdata_2['11NECKLO00THFOXA'].get_values()).reshape(len(chdata_2.index),-1),index=chdata_2.index)
combined = pd.concat((std1,std2)).std()
std1 = std1.std()
std2 = std2.std()

ti = np.logical_and(combined>std1,combined>std2)
ti = medfilt(ti,kernel_size=31).astype(bool)

ax = plt.axes()
ax.plot(t,std1,label='No-slip',color='#e66101')
ax.plot(t,std2,label='Slip',color='#5e3c99')
ax.plot(t,combined,label='Combined',color='k')
ax.axvline(x=t[ti][0],color='r',linewidth=2,label='t='+str(t[ti][0])+'s')
ax.legend()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Standard Deviation [N]')
ax.set_title('Lower Neck Fx')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.legend] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(14)