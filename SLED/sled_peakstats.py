# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:13:09 2018

@author: tangk
"""

from PMG.COM.openbook import openHDF5
from PMG.COM import arrange
from PMG.COM.get_props import *
from PMG.COM.plotfuns import *
from PMG.COM.data import import_data
import pandas as pd
from scipy.stats import anderson_ksamp
from scipy.stats import cumfreq
from PMG.read_data import read_merged
from PMG.COM import table as tb

dummy = 'Y2'
plotfigs = 1
savefigs = 0
writefiles = 0
usesmth = 0

#%%
table = tb.get('SLED')
directory = 'P:\\SLED\\Data\\'
if dummy=='Y7':
    channels = ['12CHST0000Y7DSXB',
                '12HEAD0000Y7ACRA',
                '12CHST0000Y7ACRC',
                '12PELV0000Y7ACRA',
                '12HEAD0000Y7ACXA',
                '12CHST0000Y7ACXC']
    wherepeaks = np.array(['-tive','+tive','+tive','+tive','-tive','-tive'])
    exclude = []
#    channels = ['12PELV0000Y7ACXA']
#    wherepeaks = np.array(['-tive'])
elif dummy=='Y2':
    channels = ['12HEAD0000Y2ACRA',
                '12CHST0000Y2ACRC',
                '12PELV0000Y2ACRA',
                '12HEAD0000Y2ACXA',
                '12CHST0000Y2ACXC',
                '12PELV0000Y2ACXA']
    wherepeaks = np.array(['+tive','+tive','+tive','+tive','+tive','+tive'])
    exclude = ['SE16-1012_2',
               'SE16-1014_2',
               'SE17-1015_2',
               'SE17-1016_2',
               'SE16-0399',
               'SE16-0402',
               'SE16-0395',
               'SE16-0403',
               'SE17-1025_2',
               'SE17-1026_2']
cutoff = range(100,1600)

table_y7 = table.query('DUMMY==\'' + dummy + '\'').filter(items=['SE','MODEL','INSTALL_2','SLED'])
table_y7 = table_y7.set_index('SE',drop=True)
if not(exclude==[]):
    table_y7 = table_y7.drop(exclude,axis=0)
models = np.unique(table_y7['MODEL'])
sleds = np.unique(table_y7['SLED'])
types = table_y7['INSTALL_2'].dropna().unique()

t, fulldata = import_data(directory,channels,tcns=table_y7.index)
chdata = arrange.test_ch_from_chdict(fulldata,cutoff)
t = t.get_values()[cutoff]
writename = 'C:\\Users\\tangk\\Python\\Sled_' + dummy + '_'

#%%
print('getting props...')
if not(usesmth):
    props = {'peak':arrange.arrange_by_peak(chdata.applymap(peakval)).append(pd.DataFrame(index=['cdf'])),
              'ipeak':arrange.arrange_by_peak(chdata.applymap(get_ipeak)),
              'smth':chdata.applymap(smooth_data)}
    props.update({'i2peak':arrange.arrange_by_peak(props['smth'].applymap(get_i2peak)),
                   'stats':pd.DataFrame(index=['peak','t2peak','BSp_peak','BSp_t2peak'],columns=props['peak'].columns)})
    props['t2peak'] = get_t2peak(t,props['i2peak']).append(pd.DataFrame(index=['cdf']))
    props['fwhm'] = arrange.arrange_by_peak(get_fwhm(t,chdata)).append(pd.DataFrame(index=['cdf']))
    props['tfwhm'] = arrange.arrange_by_peak(get_tfwhm(t,chdata))
else:
    props = {'smth':chdata.applymap(smooth_data)}
    props['peak'] = arrange.arrange_by_peak(props['smth'].applymap(peakval)).append(pd.DataFrame(index=['cdf']))
    props['ipeak'] = arrange.arrange_by_peak(props['smth'].applymap(get_ipeak))
    props['i2peak'] = arrange.arrange_by_peak(props['smth'].applymap(get_i2peak))
    props['t2peak'] = get_t2peak(t,props['i2peak']).append(pd.DataFrame(index=['cdf']))
    props['fwhm'] = arrange.arrange_by_peak(get_fwhm(t,props['smth'])).append(pd.DataFrame(index=['cdf']))
    props['tfwhm'] = arrange.arrange_by_peak(get_tfwhm(t,props['smth']))
#%% plot mean +/- std and distributions
if plotfigs:
    # compare across sleds for each model
    for m in models:
        for tp in types:
            for p in ['peak','t2peak', 'fwhm']:
                for i,ch in enumerate(channels):
                    x = {}
                    for s in sleds:
                        se = table_y7.query('MODEL==\''+m+'\' and SLED==\''+s+'\' and INSTALL_2==\'' + tp + '\'').index
                        if len(se)==0:
                            x[s] = [np.nan]
                        else:
                            xs = np.abs(arrange.get_values(props[p][ch][wherepeaks[i]][se].get_values().astype(float)))
                            if len(xs)>0:
                                x[s] = np.abs(arrange.get_values(props[p][ch][wherepeaks[i]][se].get_values().astype(float)))
                            else:
                                x[s] = [np.nan]
                    if np.isnan(x['new_accel']).all() and np.isnan(x['new_decel']).all() and np.isnan(x['old_accel']).all():
                        continue
                    fig = plt.figure(figsize=(5,5))
                    ax = plt.axes()
                    ax = plot_cat_nobar(ax,x)
                    ax.set_title(p + '-' + ch + '\n ' + m + ' ' + t)
                    if savefigs:
                        fig.savefig(writename + 'dot_' + ch + '_' + p + '_' + t + '_' + m + '_' + wherepeaks[i] + '.png', bbox_inches='tight')
                    plt.show()
                    plt.close(fig) 

#%% plot curves and characteristics on curve
cmap_r = np.linspace(252,63,num=len(sleds))/255
cmap_g = np.linspace(70,94,num=len(sleds))/255
cmap_b = np.linspace(107,251,num=len(sleds))/255
if plotfigs:
    for i, ch in enumerate(channels):
        n = int(np.ceil(np.sum(table_y7['SLED']=='new_accel')/10)) + int(np.ceil(np.sum(table_y7['SLED']=='new_decel')/10)) + int(np.ceil(np.sum(table_y7['SLED']=='old_accel')/10))
            
        fig, axs = plt.subplots(n,10,sharey='all',figsize=(40,5*(n))) 
        k = -1
        for j, sl in enumerate(sleds):
            se = table_y7.query('SLED==\'' + sl + '\'').index
            if len(se)%10==0:
                nnan=0
            else:
                nnan = 10-len(se)%10+1
            for s in se:
                if not(s in chdata[ch].index):
                    continue
                k = k + 1
                raw = chdata[ch][s]
                if np.isnan(raw).all():
                    continue
                smth = props['smth'][ch][s]
                ipeak = props['ipeak'][ch][wherepeaks[i]][s]
                i2peak = props['i2peak'][ch][wherepeaks[i]][s]
                if not(np.isnan(props['fwhm'][ch][wherepeaks[i]][s])):
                    fwhm_t = props['tfwhm'][ch][wherepeaks[i]][s][0::2]
                    fwhm_x = props['tfwhm'][ch][wherepeaks[i]][s][1::2]
                else:
                    fwhm_t = np.nan
                    fwhm_x = np.nan
                axs.flatten()[k].plot(t,raw,color=[cmap_r[j], cmap_g[j], cmap_b[j]],label="raw")
                if usesmth:
                    axs.flatten()[k].plot(t,smth,color='k',label="smooth")
                axs.flatten()[k].plot(t[ipeak],raw[ipeak],'.',color="r",markersize=10,label='peak')
#                axs.flatten()[k].plot(t[i2peak],raw[i2peak],'.',color='b',markersize=10,label='t2peak')
                axs.flatten()[k].plot(fwhm_t,fwhm_x,color='k',label='FWHM')
                m = table_y7['MODEL'][s]
                tp = table_y7['INSTALL_2'][s]
                axs.flatten()[k].set_title(s + '-' + sl + '\n ' + m + ' ' + tp)
            axs.flatten()[k].legend()
            k = k + nnan
        fig.suptitle(ch)
        if savefigs:
            fig.savefig(writename + '_' + ch + '_' + wherepeaks[i] + '_ts.png',bbox_inches='tight')

#%% get ratios--method B
meanprops = {}
for p in ['peak','t2peak','fwhm']:
    col1 = np.matlib.repmat(np.asarray(channels).reshape(-1,1),1,3).flatten()
    col2 = np.matlib.repmat(sleds,1,len(channels)).flatten()
    mp = pd.DataFrame(index=models,columns=[col1,col2])
    for i, ch in enumerate(channels):
        for m in models:
            for s in sleds:
                se = table_y7.query('MODEL==\''+m+'\' and SLED==\''+s+'\'').index
                if len(se)==0:
                    x = np.nan
                else:
                    x = np.abs(arrange.get_values(props[p][ch][wherepeaks[i]][se].get_values().astype(float)))
                mp.set_value(m,(ch,s),np.mean(x))
        mp[ch] = mp[ch].divide(mp[ch]['new_accel'],axis='rows')
        if plotfigs:
            fig = plt.figure(figsize=(5,5))       
            ax = plt.axes()
            ax = plot_bar(ax,mp[ch])
            ax.set_title(p + '-' + ch)
            if savefigs:
                fig.savefig(writename + p + '_' + ch + '_mean_' + wherepeaks[i] + '_bar.png', bbox_inches='tight')
            plt.show()
            plt.close(fig)
        
        log_meanprops = mp[ch].applymap(np.log)
        if writefiles:
            log_meanprops.to_csv(writename + p + '_log_meanprops_' + ch + '_' + wherepeaks[i] + '.csv')
    meanprops[p] = mp

#%% get ratios--method C
#for p in ['peak','t2peak','fwhm']:
#    meanprops = pd.DataFrame(index=models,columns=np.append(sleds,'mean_new_accel'))
#    for m in models:
#        for s in sleds:
#            se = table_y7.query('MODEL==\''+m+'\' and SLED==\''+s+'\'').index
#            if len(se)==0:
#                x = np.nan
#            else:
#                x = np.abs(arrange.get_values(props[p][ch][wherepeaks[i]][se].get_values().astype(float)))
#            if s=='new_accel':
#                meanprops.set_value(m,'mean_new_accel',np.mean(x))
#            meanprops.set_value(m,s,x/meanprops['mean_new_accel'][m])
#    barprops = {}
#    for s in sleds:
#        barprops[s] = np.concatenate(meanprops[s].dropna().get_values())
#    fig = plt.figure(figsize=(5,5))
#    ax = plt.axes()
#    ax = plot_bar(ax,barprops)
#    ax.set_title(p + '-' + ch)

#%% get ratios--method A
#for p in ['peak','t2peak','fwhm']:
#    meanprops = pd.DataFrame(index=models,columns=sleds)
#    for m in models:
#        se = table_y7.query('MODEL==\''+m+'\' and SLED==\'new_accel\'').index
#        x1 = np.abs(arrange.get_values(props[p][ch][wherepeaks[i]][se].get_values().astype(float)))
#        m_ = len(x1)
#        meanprops.set_value(m,'new_accel',[1])
#        for s in sleds:
#            if s=='new_accel':
#                continue
#            se = table_y7.query('MODEL==\''+m+'\' and SLED==\''+s+'\'').index
#            if len(se)==0:
#                meanprops.set_value(m,s,[np.nan])
#            else:
#                x2 = np.abs(arrange.get_values(props[p][ch][wherepeaks[i]][se].get_values().astype(float)))
#                n_ = len(x2)
#            div = (np.matlib.repmat(x2,m_,1)/np.matlib.repmat(x1,n_,1).T).flatten()
#            meanprops.set_value(m,s,div)
#    barprops = {}
#    for s in sleds:
#        barprops[s] = np.concatenate(meanprops[s].get_values())
#    fig = plt.figure(figsize=(5,5))
#    ax = plt.axes()
#    ax = plot_bar(ax,barprops)
#    ax.set_title(p + '-' + ch)