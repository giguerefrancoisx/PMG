# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:54:16 2018

@author: tangk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.spatial.distance import squareform, pdist


#%%
x = [range(400,600),range(500,700),range(600,800),range(700,900),range(800,1000)]
for delrange in x:
    start = str(int((list(delrange)[0]-100)/10))
    end = str(int((list(delrange)[-1]+1-100)/10))
    savename_append_at_top = 'Truncated_at_' + start + '-' + end + 'ms_HEV_vs_ICE_56_'
    runMe()

#%% calculate distance between two points relative to avg distance between all points
from scipy.spatial.distance import squareform, pdist
import statistics as stc
dist = {}
paired_dist = {}
mean_dist = {}
for mtc in metric:
    distances = pdist(linkMe,metric=mtc)
    dist[mtc] = distances
    paired_dist[mtc] = {'Escape': squareform(distances)[0,2]}
    paired_dist[mtc]['Camry'] = squareform(distances)[1,3]
    paired_dist[mtc]['Focus'] = squareform(distances)[4,12]
    paired_dist[mtc]['C-max'] = squareform(distances)[5,11]
    paired_dist[mtc]['Sonata'] = squareform(distances)[6,7]
    paired_dist[mtc]['Jetta'] = squareform(distances)[8,16]
    paired_dist[mtc]['Fusion'] = squareform(distances)[9,10]
    paired_dist[mtc]['Pacifica'] = squareform(distances)[22,23]
    paired_dist[mtc]['Accord'] = squareform(distances)[13,15]
    paired_dist[mtc]['Spark'] = squareform(distances)[14,19]
    paired_dist[mtc]['Cruze/Volt'] = squareform(distances)[17,18]
    paired_dist[mtc]['Soul'] = squareform(distances)[20,21]
    mean_dist[mtc] = stc.mean(distances)
    
    # make a bar graph for each metric
    ax = plt.axes()
    ax.bar(list(paired_dist[mtc].keys()),list(paired_dist[mtc].values()))
    ax.axhline(y=mean_dist[mtc],color='r')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.show()
# Models:
# Escape Hybrid vs. Escape: 0, 2
#Camry HEV vs. Camry: 1, 3
#Focus vs. Focus EV: 4, 12
#C-Max Hybrid vs. C-Max Plug-in: 5, 11
#Sonata HEV vs. Sonata ICE: 6, 7
#Jetta Hybrid vs. Jetta: 8, 16
#Fusion Plug-in vs. Fusion ICE: 9, 10
#Pacifica PHEV vs. Pacifica: 22, 23
#Accord vs. Accord Hybrid: 13, 15
#Spark EV vs. Spark: 14, 19
#Cruze vs. Volt: 17, 18
#Soul vs. Soul EV: 20, 21

#%% same as above but for Old vs New vehicles
from scipy.spatial.distance import squareform, pdist
import statistics as stc
dist = {}
paired_dist = {}
mean_dist = {}
for mtc in metric:
    distances = pdist(linkMe,metric=mtc)
    dist[mtc] = distances
    paired_dist[mtc] = {'MDX': squareform(distances)[0,10]}
    paired_dist[mtc]['Civic'] = squareform(distances)[1,8]
    paired_dist[mtc]['Odyssey'] = squareform(distances)[2,11]
    paired_dist[mtc]['Aveo'] = squareform(distances)[3,7]
    paired_dist[mtc]['Sentra'] = squareform(distances)[4,9]
    paired_dist[mtc]['Golf'] = squareform(distances)[5,12]
    paired_dist[mtc]['F150'] = squareform(distances)[6,13]
    mean_dist[mtc] = stc.mean(distances)
    
    # make a bar graph for each metric
    ax = plt.axes()
    ax.bar(list(paired_dist[mtc].keys()),list(paired_dist[mtc].values()))
    ax.axhline(y=mean_dist[mtc],color='r')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.show()

#%% plot euclidean distances over time
p1 = ['TC13-216','TC11-005','TC13-014','TC13-010','TC13-005','TC09-027','TC13-007','TC17-017']
p2 = ['TC12-211','TC12-012','TC13-003','TC13-012','TC13-006','TC11-008','TC14-035','TC17-208']

fig, axs = plt.subplots(3,3,sharey='all',figsize=(30,20))
for i, ax in enumerate(axs.flatten()):
    ax.plot(t,xdata[p1[i]],label=p1[i])
    ax.plot(t,xdata[p2[i]],label=p2[i])
    ax.plot(t,np.sqrt((xdata[p1[i]]-xdata[p2[i]])**2),label='difference')
    ax.legend()

#%%
fig, axs = plt.subplots(1,3,sharey='all',figsize=(20,5))
axs[0].plot(t,xdata['TC13-006'],label='TC13-006 (Sonata ICE)')
axs[0].plot(t,xdata['TC13-005'],label='TC13-005 (Sonata HEV)')
axs[0].legend()
axs[0].set_ylabel('Acceleration [g]')
axs[0].set_xlabel('Time [s]')


axs[1].plot(t,xdata['TC14-035'],label='TC13-006 (Jetta)')
axs[1].plot(t,xdata['TC13-007'],label='TC13-005 (Jetta Hybrid)')
axs[1].legend()
axs[1].set_xlabel('Time [s]')


axs[2].plot(t,xdata['TC13-012'],label='TC13-006 (Fusion ICE)')
axs[2].plot(t,xdata['TC13-010'],label='TC13-005 (Fusion Plug-in)')
axs[2].legend()
axs[2].set_xlabel('Time [s]')

#%% try quiver
from mpl_toolkits.mplot3d import Axes3D
for i in range(0,1599,50):    
    x = [np.mean(thd1['11HEAD0000THACXA'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN0100THACXC'].iloc[0,i:i+49]),
         np.mean(thd1['11CHST0000THACXC'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN1200THACXC'].iloc[0,i:i+49]),
         np.mean(thd1['11ABDOUP00THACXA'].iloc[0,i:i+49]),
         np.mean(thd1['11PELV0000THACXA'].iloc[0,i:i+49])]
    y = [np.mean(thd1['11HEAD0000THACYA'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN0100THACYC'].iloc[0,i:i+49]),
         np.mean(thd1['11CHST0000THACYC'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN1200THACYC'].iloc[0,i:i+49]),
         0,
         np.mean(thd1['11PELV0000THACYA'].iloc[0,i:i+49])]
    z = [np.mean(thd1['11HEAD0000THACZA'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN0100THACZC'].iloc[0,i:i+49]),
         np.mean(thd1['11CHST0000THACZC'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN1200THACZC'].iloc[0,i:i+49]),
         0,
         np.mean(thd1['11PELV0000THACZA'].iloc[0,i:i+49])]
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot([0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[14,13,12,8,6,4,2,0],marker='.',markersize=10)
    ax.quiver([0,0,0,0,0,0],[0,0,0,0,0,0],[14,8,6,4,2,0],x,y,z,arrow_length_ratio=1)
#ax.plot([0,0,0,0],[-1,-0.5,0.5,1],[10,10,10,10],marker='.',markersize=10)
#ax.plot([0,0.5,1,1,1],[0,1,1,1,1],[0,-2,-3,-4,-6],marker='.',markersize=10)
#ax.plot([0,0.5,1,1,1],[0,-1,-1,-1,-1],[0,-2,-3,-4,-6],marker='.',markersize=10)

#ax.quiver([0],[0],[0],[0.01],[0.01],[0.01])

#%% plot multiple channels at once
channels_to_plot = ['11HEAD0000THACXA','11SPIN0100THACXC','11CHST0000THACXC','11SPIN1200THACXC','11PELV0000THACXA']
for i in range(10):
    fig = plt.figure(figsize=(10,7))
    for ch in channels_to_plot:
        plt.plot(t,thd1[ch].iloc[i,:],label=ch)
    plt.legend()
    plt.show()

#%% plot pairs
for i in range(len(channels_to_plot)):
    for j in range(i+1,len(channels_to_plot)):
        fig, axs = plt.subplots(1,2,sharey='all',figsize=(10,5))
        axs[0].plot(t,thd1[channels_to_plot[i]].T,'b')
        axs[0].plot(t,thd1[channels_to_plot[j]].T,'g')
        axs[0].set_title('OK Belts')
        axs[1].plot(t,thd2[channels_to_plot[i]].T,'b')
        axs[1].plot(t,thd2[channels_to_plot[j]].T,'g')
        axs[1].set_title('Slip Belts')
        fig.suptitle(channels_to_plot[i] + ' (blue)' + ' vs ' + channels_to_plot[j] + ' (green)')
#%%
import plotfuns
#ch = '11FEMRLE00THFOYB'
#ch = '11CLAVLEINTHFOXA'
#ch = '11CLAVLEINTHFOZA'
#ch = '11CLAVLEOUTHFOXA'
#ch = '11CLAVLEOUTHFOZA'
#ch = '11CLAVRIINTHFOXA'
#ch = '11CLAVRIINTHFOZA'
#ch = '11CLAVRIOUTHFOXA'
ch = '11CLAVRIOUTHFOZA'
x = thd1[ch].append(thd2[ch])
f = plotfuns.plot_full(t,x,list(x.index.values),[4,5],(30,20))

#%% 2d quiver
#for i in range(0,1599,50):    
for i in range(0,299,100):
    x = np.negative([np.mean(thd1['11HEAD0000THACXA'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN0100THACXC'].iloc[0,i:i+49]),
         np.mean(thd1['11CHST0000THACXC'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN1200THACXC'].iloc[0,i:i+49]),
         np.mean(thd1['11ABDOUP00THACXA'].iloc[0,i:i+49]),
         np.mean(thd1['11PELV0000THACXA'].iloc[0,i:i+49])])
    y = [np.mean(thd1['11HEAD0000THACYA'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN0100THACYC'].iloc[0,i:i+49]),
         np.mean(thd1['11CHST0000THACYC'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN1200THACYC'].iloc[0,i:i+49]),
         0,
         np.mean(thd1['11PELV0000THACYA'].iloc[0,i:i+49])]
    z = [np.mean(thd1['11HEAD0000THACZA'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN0100THACZC'].iloc[0,i:i+49]),
         np.mean(thd1['11CHST0000THACZC'].iloc[0,i:i+49]),
         np.mean(thd1['11SPIN1200THACZC'].iloc[0,i:i+49]),
         0,
         np.mean(thd1['11PELV0000THACZA'].iloc[0,i:i+49])]
    fig, axs = plt.subplots(2,2,sharey='all',figsize=(10,10))
    
    axs[0][0].plot([0,0,0,0,0,0,0,0],[7,6,5.5,4,3,2,1,0],marker='.',markersize=10)
    axs[0][0].plot([-1,-0.5,0.5,1],[5,5,5,5],marker='.',markersize=10)
    axs[0][0].plot([0,-1.5,-1.5,-1.5,-1.5],[0,0,-2,-3,-4],marker='.',markersize=10)
    axs[0][0].plot([0,1.5,1.5,1.5,1.5],[0,0,-2,-3,-4],marker='.',markersize=10)
    axs[0][0].quiver([0,0,0,0,0,0],[7,4,3,2,1,0],y,z)
    axs[0][0].grid()
    axs[0][0].set_aspect('equal')
    
    axs[0][1].plot([0,0,0,0,0,0,0,0],[7,6,5.5,4,3,2,1,0],marker='.',markersize=10)
    axs[0][1].plot([0],[5],marker='.',markersize=10)
    axs[0][1].plot([0,-2,-2],[0,0,-4])
    axs[0][1].plot([0,-1],[0,0],marker='.',markersize=10)
    axs[0][1].plot([-2,-2,-2],[-2,-3,-4],marker='.',markersize=10)
    axs[0][1].quiver([0,0,0,0,0,0],[7,4,3,2,1,0],x,z)
    axs[0][1].grid()
    axs[0][1].set_aspect('equal')
    
    axs[1][0].plot([0,0,0,0,0,0,0,0],[7,6,5.5,4,3,2,1,0],marker='.',markersize=10)
    axs[1][0].plot([-1,-0.5,0.5,1],[5,5,5,5],marker='.',markersize=10)
    axs[1][0].plot([0,-1.5,-1.5,-1.5,-1.5],[0,0,-2,-3,-4],marker='.',markersize=10)
    axs[1][0].plot([0,1.5,1.5,1.5,1.5],[0,0,-2,-3,-4],marker='.',markersize=10)
    axs[1][0].quiver([0,0,0,0,0,0],[7,4,3,2,1,0],x,z)
    axs[1][0].grid()
    axs[1][0].set_aspect('equal')
    
    axs[1][1].plot([0,0,0,0,0,0,0,0],[7,6,5.5,4,3,2,1,0],marker='.',markersize=10)
    axs[1][1].plot([0],[5],marker='.',markersize=10)
    axs[1][1].plot([0,-2,-2],[0,0,-4])
    axs[1][1].plot([0,-1],[0,0],marker='.',markersize=10)
    axs[1][1].plot([-2,-2,-2],[-2,-3,-4],marker='.',markersize=10)
    axs[1][1].quiver([0,0,0,0,0,0],[7,4,3,2,1,0],y,z)
    axs[1][1].grid()
    axs[1][1].set_aspect('equal')


