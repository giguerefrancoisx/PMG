# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:11:44 2018

@author: giguerf
"""
import os
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from GitHub.COM.data import import_SAI, process, find_peaks

table = pd.read_excel('P:/AHEC/thortable.xlsx', sheetname='All')
tcns = table[table['CBL_BELT'].isin(['SLIP','OK'])]['CIBLE'].tolist()

SAI = os.fspath('P:/AHEC/SAI/')
sl = slice(100,1600)
channel = '11NECKLO00THFOXA'

time, data = import_SAI(SAI, channel, tcns, sl)
data = process(data, norm=False, scale=False)
peaks, times = find_peaks(data, time)
plt.close('all')
plt.plot(time, data)
plt.plot(times, peaks, '.')
slips = table[table['CBL_BELT']=='SLIP']['CIBLE'].tolist()
oks = table[table['CBL_BELT']=='OK']['CIBLE'].tolist()

#%%%
n = 1000
plt.close('all')
for param in [peaks, times]:
    slip_params = param.loc[slips].dropna()
    ok_params = param.loc[oks].dropna()
    n_samples = min([len(slip_params),len(ok_params)])//2
    slip = []
    ok = []
    for i in range(n):
        slip_sample = resample(slip_params, n_samples=n_samples)
        ok_sample = resample(ok_params, n_samples=n_samples)
        slip.extend(slip_sample)
        ok.extend(ok_sample)

    fig, axs = plt.subplots(2,2, sharex='all', sharey='row')
    axs = axs.flatten()
    slip_n, slip_bins, _ = axs[0].hist(slip)
    ok_n, ok_bins, _ = axs[1].hist(ok)
    alln, bins, _ = axs[2].hist(slip+ok)
    alln, bins, _ = axs[3].hist(slip+ok)
