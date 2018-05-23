# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:54:25 2018
get PCA metrics for thoracic deflection
@author: tangk
"""

from PMG.COM.data import import_data
from PMG.COM import arrange
import pandas as pd
import numpy as np
from PMG.COM.get_props import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import anderson_ksamp
# initialize files, params
# input
directory = 'P:\\AHEC\\Data\\THOR\\'

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

channels = ['11CHSTLEUPTHDSRB',
            '11CHSTRIUPTHDSRB',
            '11CHSTLELOTHDSRB',
            '11CHSTRILOTHDSRB']
wherepeaks = np.array(['+tive','+tive','+tive','+tive'])
cutoff = range(100,1600)

files1.sort()
files2.sort()
plotfigs = 0
savefigs = 0
writefiles = 0
bs_n_it = 5000
bs_alpha = 0.05
bs_nbin = [25,25]

#%% read data
t, fulldata_1 = import_data(directory,channels,tcns=files1)
chdata_1 = arrange.test_ch_from_chdict(fulldata_1,cutoff)
t, fulldata_2 = import_data(directory,channels,tcns=files2)
chdata_2 = arrange.test_ch_from_chdict(fulldata_2,cutoff)
t = t.get_values()[cutoff]
#%%
pcm1 = get_PC_metrics(chdata_1)
pcm2 = get_PC_metrics(chdata_2)

#%%
# scale the data
pcm = pd.concat((pcm1,pcm2))
scaled = StandardScaler().fit_transform(pcm)

# do pca
pca = PCA(n_components=1)
res = pca.fit_transform(scaled)
beta = -pca.components_.flatten()

m = pcm.mean()
s = np.std(pcm)
PCscore = (beta*pcm/s).sum(axis=1)
PCS1 = PCscore[files1].dropna()
PCS2 = PCscore[files2].dropna()

#%%
print('A-D p-value: ' + str(anderson_ksamp([PCS1,PCS2]).significance_level))


