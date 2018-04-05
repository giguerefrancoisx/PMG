# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:45:53 2018

@author: tangk
"""
import plotfuns
import numpy as np 
import bootstrap
import pandas as pd
#%%
# A-D statistic and CDF
for i in range(len(channels)):
    for j in range(i+1,len(channels)):
        for sign in ['-tive','+tive']:
            x1 = arrange.get_values(multiprops1['Dt2peak'][channels[i]][sign][channels[j]][:-1].get_values().astype(float))
            x2 = arrange.get_values(multiprops2['Dt2peak'][channels[i]][sign][channels[j]][:-1].get_values().astype(float))
            
            multiprops1['stats'].set_value((channels[j],'Dt2peak'),(channels[i],sign),anderson_ksamp([x1,x2]))
            multiprops2['stats'].set_value((channels[j],'Dt2peak'),(channels[i],sign),anderson_ksamp([x1,x2]))
            
            multiprops1['Dt2peak'].set_value((channels[j],'cdf'),(channels[i],sign),cumfreq(x1))
            multiprops2['Dt2peak'].set_value((channels[j],'cdf'),(channels[i],sign),cumfreq(x2))

#%% plot characteristics
if plotfigs:
    for i in range(len(channels)):
        for j in range(i+1,len(channels)):
            fig = plt.figure(figsize=(5,5))
            ax = plt.axes()
            x1 = arrange.get_values(multiprops1['Dt2peak'][channels[i]][wherepeaks[i]][channels[j]][:-1].get_values().astype(float))
            x2 = arrange.get_values(multiprops2['Dt2peak'][channels[i]][wherepeaks[i]][channels[j]][:-1].get_values().astype(float))
            ax = plotfuns.plot_bar(ax,['OK','Slip'],x1,x2)
            ax.set_title(channels[i]+'-'+channels[j])
            if savefigs:
                fig.savefig(writename + '_Dt2peak_' + channels[i] + '-' + channels[j] + '_bar.png',bbox_inches='tight')
            plt.show()
            plt.close(fig)
            
            #ecdf
            fig = plt.figure(figsize=(5,5))
            ax = plt.axes()
            ax = plotfuns.plot_ecdf(ax,['OK','Slip'],
                                    multiprops1['Dt2peak'][channels[i]][wherepeaks[i]][channels[j]]['cdf'],
                                    multiprops2['Dt2peak'][channels[i]][wherepeaks[i]][channels[j]]['cdf'])
            ax.set_title(channels[i]+'-'+channels[j])
            if savefigs:
                fig.savefig(writename + '_Dt2peak_' + channels[i] + '-' + channels[j] + '_ecdf.png',bbox_inches='tight')
            plt.show()
            plt.close(fig)

            # plot peaks
            if len(files1)%10==0:
                nnan = 0
            else:
                nnan = 10-len(files1)%10
            
            raw1 = chdata_1[channels[i]].append(pd.Series(np.tile([np.nan],nnan))).append(chdata_2[channels[i]])
            raw2 = chdata_1[channels[j]].append(pd.Series(np.tile([np.nan],nnan))).append(chdata_2[channels[j]])
            ipeak1 = props1['i2peak'][channels[i]][wherepeaks[i]].append(pd.Series(np.tile([np.nan],nnan))).append(props2['i2peak'][channels[i]][wherepeaks[i]])
            ipeak2 = props1['i2peak'][channels[j]][wherepeaks[i]].append(pd.Series(np.tile([np.nan],nnan))).append(props2['i2peak'][channels[j]][wherepeaks[i]])
            n1 = int(np.ceil(len(files1)/10))
            n2 = int(np.ceil(len(files2)/10))
            fig, axs = plt.subplots(n1+n2,10,sharey='all',figsize=(40,4*(n1+n2)))     
            for k, ax in enumerate(axs.flatten()[range(len(raw1))]):
                ax.set_title(raw1.index[k])
                if np.isnan(raw1[raw1.index[k]]).all():
                    continue
                elif np.isnan(raw2[raw2.index[k]]).all():
                    continue
                ax.plot(t,raw1[raw1.index[k]],label=channels[i])
                ax.plot(t[ipeak1[raw1.index[k]]],np.array(raw1[raw1.index[k]])[ipeak1[raw1.index[k]]],'.b',markersize=10,label=channels[i] + '-t2peak')
                ax.plot(t,raw2[raw2.index[k]],label=channels[j])
                ax.plot(t[ipeak2[raw2.index[k]]],np.array(raw2[raw2.index[k]])[ipeak2[raw2.index[k]]],'.r',markersize=10,label=channels[j] + '-t2peak')
            fig.suptitle(channels[i] + '(blue)-' + channels[j] + '(orange)')
            plt.show()
            if savefigs:
                fig.savefig(writename + channels[i] + '-' + channels[j] + '.png',bbox_inches='tight')
            plt.close(fig)
#%% bootstrap
for i in range(len(channels)):
    for j in range(i+1,len(channels)):
        X1 = arrange.get_values(multiprops1['Dt2peak'][channels[i]][wherepeaks[i]][channels[j]][:-1].get_values().astype(float))
        X1_resample = bootstrap.bootstrap_resample(X1,n=bs_n_it)
        L1,U1 = bootstrap.get_ci(X1,X1_resample,bs_alpha)
        b1 = bootstrap.get_bins_from_ci(np.mean(X1_resample,axis=1),[L1,U1],bs_nbin[0])
        
        X2 = arrange.get_values(multiprops2['Dt2peak'][channels[i]][wherepeaks[i]][channels[j]][:-1].get_values().astype(float))
        X2_resample = bootstrap.bootstrap_resample(X2,n=bs_n_it)
        L2,U2 = bootstrap.get_ci(X2,X2_resample,bs_alpha)
        b2 = bootstrap.get_bins_from_ci(np.mean(X2_resample,axis=1),[L2,U2],bs_nbin[1])
        
        BSp = bootstrap.test(X1,X2,nbs=bs_n_it)
        multiprops1['stats'].set_value((channels[j],'BSp_Dt2peak'),(channels[i],wherepeaks[i]),BSp)
        multiprops2['stats'].set_value((channels[j],'BSp_Dt2peak'),(channels[i],wherepeaks[i]),BSp)
        
        fig = plt.figure()
        ax = plt.axes()
        ax = plotfuns.plot_bs_distribution(ax,(np.mean(X1_resample,axis=1),b1,'OK'),(np.mean(X2_resample,axis=1),b2,'Slip'),[[L1,U1],[L2,U2]])
        ax.set_title(channels[i]+'-'+channels[j])
        plt.show()
        if savefigs:
            fig.savefig(writename + 'BSMeanDistribution_' + 'Dt2peak_' + channels[i] + '-' + channels[j] + '.png',bbox_inches='tight')
        plt.close(fig)
#%%
if writefiles:
    multiprops1['stats'].to_csv(writename + 'multichannel_stat.csv')