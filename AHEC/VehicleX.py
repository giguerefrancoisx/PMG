# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:04:52 2018

@author: giguerf
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from GitHub.COM import openbook as ob
from GitHub.COM import plotstyle as style
from GitHub.COM.fft import lowpass

readdir = os.fspath('P:/AHEC/SAI/')
savedir = os.fspath('P:/AHEC/Plots/')

subdir = 'test/'
tcns = None
#chlist = ['10SIMELE00INACXD', '10SIMERI00INACXD', '10CVEHCG0000ACXD']

time, data = ob.openbook(readdir)

left = data['10SIMELE00INACXD']
center = data['10CVEHCG0000ACXD']
right = data['10SIMERI00INACXD']

plt.close('all')

r, c = style.sqfactors(len(center.columns[:9]))
fig, axs = style.subplots(r, c, sharey = 'all', figsize=(10*c, 6.25*r))
ylim = style.ylim_no_outliers([center-right, center, right])

for i, tcn in enumerate(center.columns[:9]):
    ax = axs[i]
    try:
        ax.plot(time, center.loc[:,tcn]-right.loc[:,tcn], color='k')
        ax.plot(time, center.loc[:,tcn], color='tab:red')
        ax.plot(time, right.loc[:,tcn], color='tab:blue')
        ax.plot(time, left.loc[:,tcn], color='tab:purple')
    except KeyError:
        pass
    ax.set_title(tcn)
    ax.set_xlim(-0.01,0.3)
    ax.set_ylim(*ylim)
    ax.set_ylabel('Acceleration [g]')
    ax.set_xlabel('Time [s]')

plt.tight_layout(rect=[0,0,1,0.92])
#plt.savefig(savedir+subdir+'VehicleX.png', dpi=200)
#plt.close('all')
