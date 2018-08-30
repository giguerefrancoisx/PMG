# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:55:11 2018

@author: tangk
"""
import read_data, arrange
import pandas as pd
import numpy as np
from get_props import *
from smooth import smooth_peaks
# initialize files, params
# input
dir1 = 'P:\\AHEC\\Data\\THOR\\'

# 48 vehicle to vehicle
# OK
files1 = ['TC14-015',
          'TC15-024',
          'TC16-019',
          'TC16-205',
          'TC17-017',
          'TC17-025',
          'TC17-028',
          'TC17-029',
          'TC17-033',
          'TC17-206',
          'TC17-208',
          'TC17-211']
# slip
files2 = ['TC08-107',
          'TC09-027',
          'TC11-008',
          'TC12-003',
          'TC13-007',
          'TC13-035',
          'TC13-036',
          'TC13-119',
          'TC14-035',
          'TC14-141',
          'TC14-174',
          'TC14-502',
          'TC15-155',
          'TC15-162',
          'TC15-163',
          'TC16-016',
          'TC16-125',
          'TC17-012',
          'TC17-030',
          'TC17-031',
          'TC17-034',
          'TC17-201',
          'TC17-203',
          'TC17-209',
          'TC17-210',
          'TC17-212',
          'TC17-505']
writename = 'THOR_48_veh2veh_'

## full sample vehicle to vehicle
# OK
#files1 = ['TC14-015',
#          'TC15-024',
#          'TC16-019',
#          'TC16-205',
#          'TC17-017',
#          'TC17-025',
#          'TC17-028',
#          'TC17-029',
#          'TC17-033',
#          'TC17-206',
#          'TC17-208',
#          'TC17-211',
#          'TC14-220']
## slip
#files2 = ['TC08-107',
#          'TC09-027',
#          'TC11-008',
#          'TC12-003',
#          'TC13-007',
#          'TC13-035',
#          'TC13-036',
#          'TC13-119',
#          'TC14-035',
#          'TC14-141',
#          'TC14-174',
#          'TC14-502',
#          'TC15-155',
#          'TC15-162',
#          'TC15-163',
#          'TC16-016',
#          'TC16-125',
#          'TC17-012',
#          'TC17-030',
#          'TC17-031',
#          'TC17-034',
#          'TC17-201',
#          'TC17-203',
#          'TC17-209',
#          'TC17-210',
#          'TC17-212',
#          'TC17-505',
#          'TC14-218',
#          'TC14-231',
#          'TC12-218',
#          'TC13-217',
#          'TC14-012',
#          'TC11-234',
#          'TC15-208',
#          'TC11-233']
#writename = 'THOR_Full_Sample_veh2veh_'

## full sample barrier
#files1 = ['TC16-013',
#          'TC12-217',
#          'TC14-016',
#          'TC14-139',
#          'TC14-175',
#          'TC14-180',
#          'TC14-221',
#          'TC14-233',
#          'TC14-241',
#          'TC16-129',
#          'TC17-204',
#          'TC17-205',
#          'TC12-501',
#          'TC15-131']
#files2 = ['TC15-102',
#          'TC12-004',
#          'TC15-113',
#          'TC16-132']
#writename = 'THOR_Full_Sample_barrier_'

# 48 barrier
#files1 = ['TC12-217',
#          'TC14-016',
#          'TC14-139',
#          'TC14-175',
#          'TC14-180',
#          'TC14-221',
#          'TC14-233',
#          'TC14-241',
#          'TC16-129',
#          'TC17-204',
#          'TC17-205',]
#files2 = ['TC12-004']
#writename = 'THOR_48_barrier_'

#channels = ['11HEAD0000THACXA',
#          '11SPIN0100THACXC',
#          '11CHST0000THACXC',
#          '11SPIN1200THACXC',
#          '11PELV0000THACXA']
#wherepeaks = np.array(['-tive','-tive','-tive','-tive','-tive']) #0: local min; 1: local max
#cutoff = range(100,1600)

# head y, spine t1 y, chest y, spine t12 y, pelvis y, thsp avx, thsp avz
#channels = ['11CHSTLEUPTHDSXB',
#          '11CHSTRILOTHDSXB',
#          '11NECKLO00THFOXA',
#          '11NECKLO00THFOYA',
#          '11FEMRLE00THFOZB',
#          '11SPIN0100THACXC',
#          '11SPIN0100THACYC',
#          '11CHST0000THACXC',
#          '11CHST0000THACYC',
#          '11SPIN1200THACYC',
#          '11PELV0000THACYA',
#          '11THSP0100THAVXA',
#          '11THSP0100THAVZA']
#wherepeaks = np.array(['-tive','-tive','+tive','-tive','-tive','-tive','+tive','-tive','+tive','+tive','+tive','-tive','+tive'])
#cutoff = range(100,1600)

channels = ['11CHSTLEUPTHDSXB',
            '11CHSTRIUPTHDSXB',
            '11CHSTLELOTHDSXB',
            '11CHSTRILOTHDSXB']
wherepeaks = np.array(['-tive','-tive','-tive','-tive'])
cutoff = range(100,1600)

#channels = ['11CLAVLEINTHFOXA','11CLAVLEOUTHFOXA']
#wherepeaks = np.array(['-tive','-tive'])
#cutoff = range(100,1100)

#channels = ['10CVEHCG0000ACXD','11PELV0000THACXA','11SEBE0000B6FO0D']
#wherepeaks = np.array(['-tive','-tive','+tive'])
#cutoff = range(100,1600)

#channels = ['11NECKLO00THFOXA','11NECKLO00THMOYB','11NECKUP00THMOYB']
#wherepeaks = np.array(['+tive','+tive','+tive'])
#cutoff = range(100,1200)
#
#channels = ['11SPIN0100THACXC','11SPIN1200THACXC','11SEBE0000B3FO0D','11CHST0000THACXC']
#cutoff = range(100,1600)

files1.sort()
files2.sort()
plotfigs = 0
savefigs = 0
writefiles = 0
bs_n_it = 5000
bs_alpha = 0.05
bs_nbin = [25,25]

#%% read data
full_data_1 = read_data.read_merged(dir1,files1)
full_data_2 = read_data.read_merged(dir1,files2)

t = full_data_1[files1[0]].iloc[cutoff,0].get_values().flatten()

#%% get channels and props
chdata_1 = arrange.test_ch(full_data_1,channels,cutoff)
chdata_2 = arrange.test_ch(full_data_2,channels,cutoff)
#%%
props1 = {'peak':arrange.arrange_by_peak(chdata_1.applymap(peakval)).append(pd.DataFrame(index=['cdf'])),
          'ipeak':arrange.arrange_by_peak(chdata_1.applymap(get_ipeak)),
          'smth':chdata_1.applymap(smooth_data)}
#          'smth': chdata_1.applymap(smooth_peaks)}
props1.update({'i2peak':arrange.arrange_by_peak(props1['smth'].applymap(get_i2peak)),
#props1.update({'i2peak':arrange.arrange_by_peak(props1['smth'].applymap(get_ipeak)),
               'stats':pd.DataFrame(index=['peak','t2peak','BSp_peak','BSp_t2peak'],columns=props1['peak'].columns)})
props1['t2peak'] = get_t2peak(t,props1['i2peak']).append(pd.DataFrame(index=['cdf']))
#props1['PC_metrics'] = get_PC_metrics(chdata_1)
props2 = {'peak':arrange.arrange_by_peak(chdata_2.applymap(peakval)).append(pd.DataFrame(index=['cdf'])),
          'ipeak':arrange.arrange_by_peak(chdata_2.applymap(get_ipeak)),
          'smth':chdata_2.applymap(smooth_data)}
#          'smth': chdata_2.applymap(smooth_peaks)}
props2.update({'i2peak':arrange.arrange_by_peak(props2['smth'].applymap(get_i2peak)),
#props2.update({'i2peak':arrange.arrange_by_peak(props2['smth'].applymap(get_ipeak)),
               'stats':pd.DataFrame(index=['peak','t2peak','BSp_peak','BSp_t2peak'],columns=props2['peak'].columns)})
props2['t2peak'] = get_t2peak(t,props2['i2peak']).append(pd.DataFrame(index=['cdf']))
#props2['PC_metrics'] = get_PC_metrics(chdata_2)
# multiple channels
multiprops1 = {'Dt2peak':get_Dt2peak(props1)}
multiprops1['stats'] = pd.DataFrame(index=pd.MultiIndex.from_product([channels,['Dt2peak','BSp_Dt2peak']]),columns=multiprops1['Dt2peak'].columns)
multiprops2 = {'Dt2peak':get_Dt2peak(props2)}
multiprops2['stats'] = pd.DataFrame(index=pd.MultiIndex.from_product([channels,['Dt2peak','BSp_Dt2peak']]),columns=multiprops2['Dt2peak'].columns)
