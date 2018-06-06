# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:54:25 2018
get PCA metrics for thoracic deflection
@author: tangk
"""

from PMG.COM.data import import_data
import pandas as pd
import numpy as np
from PMG.COM.get_props import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import anderson_ksamp
from PMG.COM import bootstrap, plotfuns, arrange
# initialize files, params
# input
directory = 'P:\\AHEC\\Data\\THOR\\'

# 48 vehicle to vehicle
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
#          'TC17-211']
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
#          'TC17-505']
#writename = 'THOR_48_veh2veh_'

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

# Sled
files1 = ['TC58-278_2',
          'TC58-278_3']
files2 = ['TC58-278_4']

channels = ['11PCA00000THAV0B']
wherepeaks = np.array(['+tive'])
cutoff = range(1)

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

#%% get values
def get_values(data):
    return data[0]

chdata_1 = chdata_1.applymap(get_values)
chdata_2 = chdata_2.applymap(get_values)

print('No-slip:')
print('Mean: ' + str(chdata_1.mean()))
print('Std: ' + str(chdata_1.std()))
print('Slip:')
print('Mean: ' + str(chdata_2.mean()))
print('Std: ' + str(chdata_2.std()))

print('A-D p-value: ' + str(anderson_ksamp([chdata_1.values.flatten(),chdata_2.values.flatten()]).significance_level))

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

#%%
X1 = chdata_1.values.flatten()
X1_resample = bootstrap.bootstrap_resample(X1,n=bs_n_it)

# centered bootstrap percentile confidence interval
L1, U1 = bootstrap.get_ci(X1,X1_resample,bs_alpha)
b1 = bootstrap.get_bins_from_ci(np.mean(X1_resample,axis=1),[L1,U1],bs_nbin[0])

X2 = chdata_2.values.flatten()
X2_resample = bootstrap.bootstrap_resample(X2,n=bs_n_it)
    
L2,U2 = bootstrap.get_ci(X2,X2_resample,bs_alpha)
b2 = bootstrap.get_bins_from_ci(np.mean(X2_resample,axis=1),[L2,U2],bs_nbin[1])

BSp = bootstrap.test(X1,X2,nbs=bs_n_it)
display('Bootstrap p-value: ' + str(BSp))

fig = plt.figure()
ax = plt.axes()
ax = plotfuns.plot_bs_distribution(ax,(np.mean(X1_resample,axis=1),b1,'OK'),(np.mean(X2_resample,axis=1),b2,'Slip'),[[L1,U1],[L2,U2]])
ax.set_title(p + '-' + ch)
plt.show()

fig = plt.figure(figsize=(3,3))
ax = plt.axes()
mean1 = np.mean(np.mean(X1_resample,axis=1))
mean2 = np.mean(np.mean(X2_resample,axis=1))
ax.bar(['OK','Slip'],[mean1,mean2],0.7,color=[0.4, 0.4, 0.4])
ax.errorbar(['OK','Slip'],[mean1,mean2],[[mean1-L1,mean2-L2],[U1-mean1,U2-mean2]],ecolor='black',capsize=5,linestyle='none')
#ax.set_title(p + '-' + ch)
plt.rc('font',size=12)
plt.show()
