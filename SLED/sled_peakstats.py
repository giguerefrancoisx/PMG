# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:13:09 2018

@author: tangk
"""

from PMG.COM.openbook import openHDF5
from PMG.COM import arrange
from PMG.COM.get_props import *
from PMG.COM.plotfuns import plot_full_2
from PMG.COM.data import import_data
import pandas as pd
from scipy.stats import anderson_ksamp
from scipy.stats import cumfreq

directory = 'P:/SLED/Data/'
channels = ['12HEAD0000Y7ACXA',
            '12CHST0000Y7ACXC',
            '12PELV0000Y7ACXA']
wherepeaks = np.array(['-tive','-tive','-tive'])
cutoff = range(100,1600)
gr2 = ['SE16-0208',
       'SE16-0209',
       'SE16-0210',
       'SE16-0338',
       'SE16-0340',
       'SE16-0344',
       'SE16-0382',
       'SE16-0384',
       'SE16-0385',
       'SE16-0389',
       'SE16-0391',
       'SE16-0392',
       'SE16-0393',
       'SE16-0413']
gr3 = ['SE15-0796',
       'SE16-0212',
       'SE16-0336',
       'SE16-0339',
       'SE16-0386']
gr5 = ['SE15-0795',
       'SE15-0797',
       'SE16-0206',
       'SE16-0207',
       'SE16-0207_2',
       'SE16-0337',
       'SE16-0388',
       'SE16-0412',
       'SE16-0414',
       'SE16-0416',
       'SE16-0417_2']
#t, fulldata = openHDF5(directory, channels)
t, fulldata_3 = import_data(directory,channels,tcns=gr3)
t, fulldata_25 = import_data(directory,channels,tcns=gr2+gr5)
chdata_1 = arrange.test_ch_from_chdict(fulldata_3,cutoff)
chdata_2 = arrange.test_ch_from_chdict(fulldata_25,cutoff)
t = t.get_values()[cutoff]

plotfigs = 1
savefigs = 1
writefiles = 1
bs_n_it = 5000
bs_alpha = 0.05
bs_nbin = [25,25]
savename = 'C:\\Users\\tangk\\Python\\Sled_Y7_'

#%%
props1 = {'peak':arrange.arrange_by_peak(chdata_1.applymap(peakval)).append(pd.DataFrame(index=['cdf'])),
          'ipeak':arrange.arrange_by_peak(chdata_1.applymap(get_ipeak)),
          'smth':chdata_1.applymap(smooth_data)}
props1.update({'i2peak':arrange.arrange_by_peak(props1['smth'].applymap(get_i2peak)),
               'stats':pd.DataFrame(index=['peak','t2peak','BSp_peak','BSp_t2peak'],columns=props1['peak'].columns)})
props1['t2peak'] = get_t2peak(t,props1['i2peak']).append(pd.DataFrame(index=['cdf']))
props2 = {'peak':arrange.arrange_by_peak(chdata_2.applymap(peakval)).append(pd.DataFrame(index=['cdf'])),
          'ipeak':arrange.arrange_by_peak(chdata_2.applymap(get_ipeak)),
          'smth':chdata_2.applymap(smooth_data)}
props2.update({'i2peak':arrange.arrange_by_peak(props2['smth'].applymap(get_i2peak)),
               'stats':pd.DataFrame(index=['peak','t2peak','BSp_peak','BSp_t2peak'],columns=props2['peak'].columns)})
props2['t2peak'] = get_t2peak(t,props2['i2peak']).append(pd.DataFrame(index=['cdf']))


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

#%%
for ch in channels:
    plot_full_2(t,chdata_1[ch],chdata_2[ch])

