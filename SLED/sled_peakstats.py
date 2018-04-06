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

table = tb.get('SLED')
directory = 'P:/SLED/Data/'
channels = ['12CHST0000Y7DSXB']
wherepeaks = np.array(['-tive'])
cutoff = range(100,1600)

table_y7 = table.query('DUMMY==\'Y7\'').filter(items=['SE','MODEL','SLED'])
table_y7 = table_y7.set_index('SE',drop=True)
models = np.unique(table_y7['MODEL'])
sleds = np.unique(table_y7['SLED'])

t, fulldata = import_data(directory,channels,tcns=table_y7.index)
chdata = arrange.test_ch_from_chdict(fulldata,cutoff)
#chdata = pd.concat([table_y7,chdata],axis=1)
t = t.get_values()[cutoff]

plotfigs = 1
savefigs = 1
writefiles = 1
bs_n_it = 5000
bs_alpha = 0.05
bs_nbin = [25,25]
writename = 'C:\\Users\\tangk\\Python\\Sled_Y7_'

#%%
props = {'peak':arrange.arrange_by_peak(chdata.applymap(peakval)).append(pd.DataFrame(index=['cdf'])),
          'ipeak':arrange.arrange_by_peak(chdata.applymap(get_ipeak)),
          'smth':chdata.applymap(smooth_data)}
props.update({'i2peak':arrange.arrange_by_peak(props['smth'].applymap(get_i2peak)),
               'stats':pd.DataFrame(index=['peak','t2peak','BSp_peak','BSp_t2peak'],columns=props['peak'].columns)})
props['t2peak'] = get_t2peak(t,props['i2peak']).append(pd.DataFrame(index=['cdf']))


#%% plot mean +/- std and distributions
if plotfigs:
    # compare across sleds for each model
    for m in models:
        for p in ['peak','t2peak']:
            for i,ch in enumerate(channels):
                fig = plt.figure(figsize=(5,5))
                ax = plt.axes()
                x = {}
                for s in sleds:
                    se = table_y7.query('MODEL==\''+m+'\' and SLED==\''+s+'\'').index
                    if len(se)==0:
                        x[s] = [np.nan]
                    else:
                        x[s] = np.abs(arrange.get_values(props[p][ch][wherepeaks[i]][se].get_values().astype(float)))
                ax = plot_cat_nobar(ax,x)
                ax.set_title(p + '-' + ch + '\n ' + m)
                if savefigs:
                    fig.savefig(writename + p + '_' + ch + '_' + m + '_' + wherepeaks[i] + '_bar.png', bbox_inches='tight')
                plt.show()
                plt.close(fig) 

#%% get ratios
for p in ['peak','t2peak']:
    meanprops = pd.DataFrame(index=models,columns=sleds)
    for m in models:
        for s in sleds:
            se = table_y7.query('MODEL==\''+m+'\' and SLED==\''+s+'\'').index
            if len(se)==0:
                x = np.nan
            else:
                x = np.abs(arrange.get_values(props[p][ch][wherepeaks[i]][se].get_values().astype(float)))
            meanprops.set_value(m,s,np.mean(x))
 
    fig = plt.figure(figsize=(5,5))       
    ax = plt.axes()
    ax = plot_bar(ax,meanprops)
    ax.set_title(p + '-' + ch)
    if savefigs:
        fig.savefig(writename + p + '_' + ch + '_mean_' + wherepeaks[i] + '_bar.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(meanprops.divide(meanprops['new_accel'],axis='rows').mean())
