# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:13:09 2018

@author: tangk
"""

from PMG.COM import arrange
from PMG.COM.get_props import *
from PMG.COM.plotfuns import *
from PMG.COM.data import import_data
import pandas as pd
from PMG.COM import table as tb

dummy = 'Y2'
plotfigs = 1
savefigs = 0
writefiles = 0
usesmth = 0
install = 'INSTALL'
model = 'MODEL'
#%%
table = tb.get('SLED')
directory = 'P:\\SLED\\Data\\'
if dummy=='Y7':
    channels = ['12CHST0000Y7DSXB',
                '12HEAD0000Y7ACRA',
                '12CHST0000Y7ACRC',
                '12PELV0000Y7ACRA',
                '12HEAD0000Y7ACXA',
                '12CHST0000Y7ACXC',
                '12HEAD0000Y7ACZA',
                '12CHST0000Y7ACZC',
                '12PELV0000Y7ACZA']
    wherepeaks = np.array(['-tive','+tive','+tive','+tive','-tive','-tive','+tive','-tive','-tive'])
    exclude = ['SE16-0204']
#    channels = ['12PELV0000Y7ACXA']
    wherepeaks = np.array(['-tive'])
elif dummy=='Y2':
    channels = ['12HEAD0000Y2ACRA',
                '12CHST0000Y2ACRC',
                '12PELV0000Y2ACRA',
                '12HEAD0000Y2ACXA',
                '12CHST0000Y2ACXC',
                '12PELV0000Y2ACXA',
                '12HEAD0000Y2ACZA',
                '12CHST0000Y2ACZC',
                '12PELV0000Y2ACZA']
    wherepeaks = np.array(['+tive','+tive','+tive','+tive','+tive','+tive','+tive','+tive','+tive'])
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

table_y7 = table.query('DUMMY==\'' + dummy + '\'').filter(items=['SE'] + [model] + [install] + ['SLED'])
table_y7 = table_y7.set_index('SE',drop=True)
if not(exclude==[]):
    table_y7 = table_y7.drop(exclude,axis=0)
models = np.unique(table_y7[model])
sleds = np.unique(table_y7['SLED'])
types = table_y7[install].dropna().unique()

t, fulldata = import_data(directory,channels,tcns=table_y7.index)
chdata = arrange.test_ch_from_chdict(fulldata,cutoff)
t = t.get_values()[cutoff]
writename = 'C:\\Users\\tangk\\Python\\Sled_' + dummy + '_'


cmap_r = np.linspace(252,63,num=len(sleds))/255
cmap_g = np.linspace(70,94,num=len(sleds))/255
cmap_b = np.linspace(107,251,num=len(sleds))/255
#%%
print('getting props...')
if not(usesmth):
    props = {'peak':arrange.arrange_by_peak(chdata.applymap(peakval)),
              'ipeak':arrange.arrange_by_peak(chdata.applymap(get_ipeak)),
              'smth':chdata.applymap(smooth_data)}
    props.update({'i2peak':arrange.arrange_by_peak(props['smth'].applymap(get_i2peak)),
                   'stats':pd.DataFrame(index=['peak','t2peak','BSp_peak','BSp_t2peak'],columns=props['peak'].columns)})
    props['t2peak'] = get_t2peak(t,props['i2peak'])
    props['fwhm'] = arrange.arrange_by_peak(get_fwhm(t,chdata))
    props['tfwhm'] = arrange.arrange_by_peak(get_tfwhm(t,chdata))
else:
    props = {'smth':chdata.applymap(smooth_data)}
    props['peak'] = arrange.arrange_by_peak(props['smth'].applymap(peakval))
    props['ipeak'] = arrange.arrange_by_peak(props['smth'].applymap(get_ipeak))
    props['i2peak'] = arrange.arrange_by_peak(props['smth'].applymap(get_i2peak))
    props['t2peak'] = get_t2peak(t,props['i2peak'])
    props['fwhm'] = arrange.arrange_by_peak(get_fwhm(t,props['smth']))
    props['tfwhm'] = arrange.arrange_by_peak(get_tfwhm(t,props['smth']))

multiprops = {'Dpeak':get_Dpeak(props)}
#%% plot mean +/- std and distributions
if plotfigs:
    # compare across sleds for each model
    for m in models:
        for tp in types:
            for p in ['peak','t2peak', 'fwhm']:
                for i,ch in enumerate(channels):
                    x = {}
                    for s in sleds:
                        se = table_y7.query(model + '==\''+m+'\' and SLED==\''+s+'\' and ' + install + '==\'' + tp + '\'').index
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
                    ax.set_title(p + '-' + ch + '\n ' + m + ' ' + tp)
                    if savefigs:
                        fig.savefig(writename + 'dot_' + ch + '_' + p + '_' + tp + '_' + m + '_' + wherepeaks[i] + '.png', bbox_inches='tight')
                    plt.show()
                    plt.close(fig) 

#%% plot overlays
#for j, ch in enumerate(channels[0:1]):
#    for m in models:
for m in models:
    for tp in types:
        for j, ch in enumerate(channels):
            # find tests with the given model and installation
            se_new = np.asarray(table_y7.query(model + '==\'' + m + '\' and ' + install + '==\'' + tp + '\' and SLED==\'new_accel\'').index)
            se_old = np.asarray(table_y7.query(model + '==\'' + m + '\' and ' + install + '==\'' + tp + '\' and SLED==\'old_accel\'').index)
            se_decel = np.asarray(table_y7.query(model + '==\'' + m + '\' and ' + install + '==\'' + tp + '\' and SLED==\'new_decel\'').index)
            
            se_new = se_new[[~np.isnan(chdata[ch][se_new][i]).all() for i in range(len(se_new))]]
            se_old = se_old[[~np.isnan(chdata[ch][se_old][i]).all() for i in range(len(se_old))]]
            se_decel = se_decel[[~np.isnan(chdata[ch][se_decel][i]).all() for i in range(len(se_decel))]]
            # number of tests on new_accel > 1 
#            if len(se_new) == 0 or len(se_old)==0:
#                continue
            if len(se_new) == 0 or len(se_decel)==0:
                continue
            
            for s in se_new:
                plt.plot(t,chdata[ch][s],color=[cmap_r[0], cmap_g[0], cmap_b[0]],label='new_accel ' + s)
#                plt.plot(t[props['ipeak'][ch][wherepeaks[j]][s]],props['peak'][ch][wherepeaks[j]][s],'.',color='r',markersize=10)
#                plt.plot(props['t2peak'][ch][wherepeaks[j]][s],chdata[ch][s][props['i2peak'][ch][wherepeaks[j]][s]],'*',color='b',markersize=10)
#            for s in se_old:
#                plt.plot(t,chdata[ch][s],color=[cmap_r[2], cmap_g[2], cmap_b[2]],label='old_accel ' + s)
#                plt.plot(t[props['ipeak'][ch][wherepeaks[j]][s]],props['peak'][ch][wherepeaks[j]][s],'.',color='r',markersize=10)
#                plt.plot(props['t2peak'][ch][wherepeaks[j]][s],chdata[ch][s][props['i2peak'][ch][wherepeaks[j]][s]],'*',color='b',markersize=10)
            for s in se_decel:
                plt.plot(t,chdata[ch][s],'g')
#            plt.legend()
#            plt.ylim((-70,20))
            plt.title(ch + ' ' + m + ' ' + tp)
            if savefigs:
                plt.savefig(writename + 'overlay_' + ch + '_' + tp + '_' + m + '.png',bbox_inches='tight')
            plt.show()

#%% get differences--method B
meanprops = {}
for p in ['peak','t2peak','fwhm']:
    
    if p=='peak':
        def apply_round(num):
            return np.round_(num,decimals=1)
    else:
        def apply_round(num):
            return np.round_(num,decimals=3)
    
    col1 = np.matlib.repmat(np.asarray(channels).reshape(-1,1),1,3).flatten()
    col2 = np.matlib.repmat(sleds,1,len(channels)).flatten()
    mp = pd.DataFrame(index=models,columns=[col1,col2])
    for i, ch in enumerate(channels):
        for tp in types:
            for m in models:
                for s in sleds:
                    se = table_y7.query(model + '==\''+m+'\' and SLED==\''+s+'\' and ' + install + '==\'' + tp + '\'').index
                    if len(se)==0:
                        x = np.nan
                    else:
                        x = np.abs(arrange.get_values(props[p][ch][wherepeaks[i]][se].get_values().astype(float)))
                    mp.set_value(m,(ch,s),np.mean(x))
            mp[ch] = mp[ch].subtract(mp[ch]['new_accel'],axis='rows')
            display_table = mp[ch][['new_decel','old_accel']].dropna(how='all').dropna(how='all',axis=1)
            if plotfigs:
                fig = plt.figure(figsize=(5,5))       
                ax = plt.axes()
                ax = plot_bar(ax,display_table)
                ax.set_title(p + '-' + ch + ' ' + tp)
                ax.xaxis.set_ticklabels([])
                plottable = ax.table(cellText=display_table.applymap(apply_round).get_values(),rowLabels=display_table.index,colLabels=display_table.columns)
                plottable.scale(1,1.5)
                if savefigs:
                    fig.savefig(writename + 'bar_' + ch + '_' + p + '_' + tp + '_' + wherepeaks[i] + '_bar.png', bbox_inches='tight')
                plt.show()
                plt.close(fig)
            
            if writefiles:
                mp[ch].to_csv(writename + 'delta_meanprops_' + ch + '_' + p + '_' + tp + '_' + wherepeaks[i] + '.csv')
#            display(mp[ch][['new_decel','old_accel']].dropna(how='all'))
#            display(p + ' ' + ch + ' ' + tp + ':')
#            display('mean: ' + str(np.nanmean(mp[ch]['old_accel'].astype(float))))
#            display('std: ' + str(np.nanstd(mp[ch]['old_accel'].astype(float))))

#%% get ratios--method B
#meanprops = {}
#for p in ['peak','t2peak','fwhm']:
#    col1 = np.matlib.repmat(np.asarray(channels).reshape(-1,1),1,3).flatten()
#    col2 = np.matlib.repmat(sleds,1,len(channels)).flatten()
#    mp = pd.DataFrame(index=models,columns=[col1,col2])
#    for i, ch in enumerate(channels):
#        for tp in types:
#            for m in models:
#                for s in sleds:
#                    se = table_y7.query(model + '==\''+m+'\' and SLED==\''+s+'\' and ' + install + '==\'' + tp + '\'').index
#                    if len(se)==0:
#                        x = np.nan
#                    else:
#                        x = np.abs(arrange.get_values(props[p][ch][wherepeaks[i]][se].get_values().astype(float)))
#                    mp.set_value(m,(ch,s),np.mean(x))
#            mp[ch] = mp[ch].divide(mp[ch]['new_accel'],axis='rows')
#            if plotfigs:
#                fig = plt.figure(figsize=(5,5))       
#                ax = plt.axes()
#                ax = plot_bar(ax,mp[ch])
#                ax.set_title(p + '-' + ch + ' ' + tp)
#                if savefigs:
#                    fig.savefig(writename + 'bar_' + ch + '_' + p + '_' + tp + '_' + wherepeaks[i] + '_bar.png', bbox_inches='tight')
#                plt.show()
#                plt.close(fig)
#            
#            log_meanprops = mp[ch].applymap(np.log)
#            if writefiles:
#                log_meanprops.to_csv(writename + 'log_meanprops_' + ch + '_' + p + '_' + tp + '_' + wherepeaks[i] + '.csv')
#                mp[ch].to_csv(writename + 'meanprops_' + ch + '_' + p + '_' + tp + '_' + wherepeaks[i] + '.csv')
##            display(mp[ch])
#            display(p + ' ' + ch + ' ' + tp + ':')
#            display('mean: ' + str(np.nanmean(mp[ch]['old_accel'].astype(float))))
#            display('std: ' + str(np.nanstd(mp[ch]['old_accel'].astype(float))))

