# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:40:27 2017

@author: giguerf
"""

import os
import matplotlib.pyplot as plt
#import scipy.signal
#import math
#import pandas
#from lookup_pop import lookup_pop
from collections import OrderedDict
from GitHub.COM import openbook as ob
from GitHub.COM import plotstyle as style
from GitHub.COM import get_peak as gp
#import plotstyle as style
    
plt.close('all')
directory = os.fspath('P:/BOOSTER/SAI')

time, gendict, cutdict, g, c = ob.gencut(directory, '')
#channel = '14LUSP0000Y7FOXA'
channel = '14CHST0000Y7ACXC'
#traces = gendict[channel]
#ctraces = cutdict[channel]
#%%
keys = list(set(g.Marque))
colors = style.colordict(keys)

title = ['off48', 'off56', 'mur48', 'mur56']
#plt.rcParams["axes.titlesize"] = 8
ylabel = 'Force (Z-direction) [N]'

data = ob.grids(gendict, [channel])
plotdata = data[''][channel]
#%%
plt.close('all')
plt.figure('Grid')
for j, place in enumerate(title):

    plt.subplot(2,2,j+1)
    
    lines = [plt.plot(time, plotdata[place][tcn], color = colors[list(g[g.ALL.isin([tcn])].Marque)[0]])[0] for tcn in plotdata[place].columns]
    names = g[g.ALL.isin(plotdata[place].columns)].iloc[:,3:5]
#    linelabels = [str(m)+' - '+str(n) for m,n in zip(list(names['Marque']), list(names['Modele']))]
    linelabels = list(names['Marque'])
    
    plt.xlim([0,0.2])
    plt.title(title[j])
    plt.ylabel(ylabel)
    plt.xlabel('Time [s]')

    by_label = OrderedDict(zip(linelabels, lines))
    plt.legend(by_label.values(), by_label.keys(), loc = 4)
plt.subplots_adjust(top=0.961,bottom=0.065,left=0.065,right=0.968,hspace=0.217,wspace=0.292)
#again, aligned by start time
plt.figure('Aligned')
for j, place in enumerate(title):
    peaks = {}
    plt.subplot(2,2,j+1)
    for tcn in plotdata[place]:
        series = plotdata[place][tcn]
        peaks[tcn] = p = gp.get_peak(time, series)
#        series = scipy.signal.savgol_filter(plotdata[place][tcn], 91, 5)
        plt.plot(time+0.05-p.t0, series, color = colors[list(g[g.ALL.isin([tcn])].Marque)[0]], label = g[g.ALL == tcn].Marque.tolist()[0])
#    names = g[g.ALL.isin(plotdata[j].columns)].iloc[:,3:5]
#    linelabels = [str(m)+' - '+str(n) for m,n in zip(list(names['Marque']), list(names['Modele']))]
#    linelabels = list(names['Marque'])
    
    plt.xlim([0,0.2])
    plt.title(title[j])
    plt.ylabel(ylabel)
    plt.xlabel('Time [s]')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 4)
plt.subplots_adjust(top=0.961,bottom=0.065,left=0.065,right=0.968,hspace=0.217,wspace=0.292)
#%% - Align by start time, peak demo
#plt.close('all')
#peaks = {}
#for tcn in plotdata[0]:
#    series = plotdata[0][tcn]
#    peaks[tcn] = gp.get_peak(time, series)
#    
#    plt.subplot(221)
#    plt.xlim([0,0.2])
#    plt.plot(time, series)
#    plt.plot([peaks[tcn].t0], [peaks[tcn].start], '.')
#    plt.axvline(x=peaks[tcn].t0)
#    plt.axvline(x=peaks[tcn].tp)
#    
#    plt.subplot(222)
#    plt.xlim([0,0.2])
#    plt.plot(time+0.05-peaks[tcn].t0, series)
#    plt.plot([0.05],[peaks[tcn].start], '.')
#    plt.axvline(x=0.05)
#    
#    plt.subplot(224)
#    plt.xlim([0,0.2])
#    plt.plot(time+0.05-peaks[tcn].tp, series)
#    plt.plot([0.05],[peaks[tcn].peak], '.')
#    plt.axvline(x=0.05)
