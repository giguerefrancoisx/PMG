# -*- coding: utf-8 -*-
"""
TEMP FILE
    Investigating plot types
    
Created on Tue Oct 24 16:59:06 2017

@author: giguerf
"""

import os
import matplotlib.pyplot as plt
import math
import pandas
#import scipy
#import numpy as np
from textwrap import wrap
from lookup_pop import lookup_pop
from GitHub.COM import get_peak as gp
#from get_peak import get_peak
#from foreign_code import useful_function

#TCNs = ['TC12-003_14','TC17-206_14','TC14-174_14','TC17-110_16','TC15-210_14','TC12-212_14','TC14-503_14','TC17-201_14','TC15-209_14','TC16-172_14','TC14-127_16']
#TCNs = ['TC08-035_14', 'TC10-229_14', 'TC11-008_14', 'TC12-003_14',
#       'TC12-212_14', 'TC12-212_16', 'TC13-207_14', 'TC13-207_16',
#       'TC13-215_16', 'TC13-217_14', 'TC14-009_14', 'TC14-016_14',
#       'TC14-127_16', 'TC14-139_16', 'TC14-143_14', 'TC14-163_16',
#       'TC14-174_14', 'TC14-175_16', 'TC14-227_14', 'TC14-238_14',
#       'TC14-503_14', 'TC15-004_14', 'TC15-102_14', 'TC15-123_14',
#       'TC15-138_14', 'TC15-163_14', 'TC15-206_14', 'TC15-209_14',
#       'TC15-210_14', 'TC16-104_14', 'TC16-104_16', 'TC16-111_14',
#       'TC16-111_16', 'TC16-113_14', 'TC16-113_16', 'TC16-172_14',
#       'TC16-205_14', 'TC17-109_14', 'TC17-110_16', 'TC17-112_14',
#       'TC17-113_14', 'TC17-113_16', 'TC17-116_16', 'TC17-201_14',
#       'TC17-206_14', 'TC17-210_14', 'TC17-211_14']
TCNs = ['TC10-229_14','TC12-212_14','TC12-212_16','TC13-207_14','TC13-207_16',
        'TC14-139_16','TC14-503_14','TC15-102_14','TC15-206_14','TC15-209_14',
        'TC15-210_14','TC16-111_14','TC16-111_16','TC16-113_16','TC16-172_14',
        'TC16-205_14','TC17-110_16','TC17-113_16','TC17-116_16','TC17-210_14']


plt.close('all')
fulldata = []
directory = os.fspath('P:/BOOSTER/SAI')
colors = {'Chest':'tab:blue','Pelvis':'tab:green','Chest2':'tab:red','Pelvis2':'tab:purple'}
sets = {}
fulldata = {}
places = ['Chest','Pelvis']
i = 0
for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
        chdata = pandas.read_excel(directory+'/'+filename)
        fulldata[places[i]] = chdata #fulldata variable is for debugging
        i = i+1
tot = pandas.concat(list(fulldata.values()), axis = 1)
time = chdata.iloc[:,0] #rename time channel for legibility
#%% Plot
if len(TCNs) >= 9:
    TCNgroups = [TCNs[x:x+9] for x in range(0, len(TCNs), 9)]
else:
    TCNgroups = [TCNs]
k = 1

for TCNs in TCNgroups:
    plt.figure('A%d'% k)
    i = 1
    for TCN in TCNs:
        chest = fulldata['Chest'][TCN].rolling(window=30,center=False).mean().shift(-15) 
        pelvis = fulldata['Pelvis'][TCN].rolling(window=30,center=False).mean().shift(-15) 
        [g, c] = lookup_pop([TCN],'')
        
        points = [[[],[]],[[],[]]]
        [t0, tm, st, mp] = gp.get_peak(time, chest)
        points[0][0].extend([t0,tm])
        points[0][1].extend([st,mp])
                
        [t0, tm, st, mp] = gp.get_peak(time, pelvis)
        points[1][0].extend([t0,tm])
        points[1][1].extend([st,mp])
        
        bounds = (0,0.2,math.floor(min(tot.min())),math.ceil(max(tot.max())))
        bounds = (0,0.2,-140,30)
        title = ', '.join(map(str, g.loc[0].tolist()[0:5]))
        title = '\n'.join(wrap(title, 31))

        plt.rcParams["axes.titlesize"] = 8
        ylabel = 'Acceleration (X-direction) [g]'
        
        plt.subplot(3,3,i)
        plt.plot(time, chest, label = 'chest')    
        plt.plot(time, pelvis, label = 'pelvis')
        plt.scatter(points[0][0], points[0][1], marker = '+')
        plt.scatter(points[1][0], points[1][1], marker = '+')
        plt.axis(bounds)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.legend(loc = 4)
        plt.draw()
        plt.subplots_adjust(top = 0.96, bottom = 0.06, left = 0.09, right = 0.97, hspace = 0.34, wspace = 0.46)
        i= i+1
#%% custom plot
    plt.figure('B%d'% k)
    i = 1
    for TCN in TCNs:
        chest = fulldata['Chest'][TCN].rolling(window=30,center=False).mean().shift(-15) 
        pelvis = fulldata['Pelvis'][TCN].rolling(window=30,center=False).mean().shift(-15)
        [g, c] = lookup_pop([TCN],'')
        
        points = [[[],[]],[[],[]]]
        [t0, tm, st, mp] = gp.get_peak(time, chest)
        points[0][0].extend([t0,tm])
        points[0][1].extend([st,mp])
        
        [t0, tm, st, mp] = gp.get_peak(time, pelvis)
        points[1][0].extend([t0,tm])
        points[1][1].extend([st,mp])
        
        if points[0][0][1] < points[1][0][1]: #tch < tpel
            t1 = points[0][0][1]
            t2 = points[1][0][1]
        else:
            t1 = points[1][0][1]
            t2 = points[0][0][1]
        
        title = ', '.join(map(str, g.loc[0].tolist()[1:5]))
        title = '\n'.join(wrap(title, 22))
        plt.rcParams["axes.titlesize"] = 8
        ylabel = 'Acceleration (X-direction) [g]'
        
        plt.subplot(3,3,i)
        plt.plot(chest[time<=t1], pelvis[time<=t1], label = TCN)
        plt.plot(chest[(time>=t1)&(time<=t2)], pelvis[(time>=t1)&(time<=t2)], label = TCN)
        plt.plot([0,-100],[0,-100])
        plt.scatter(points[0][1], points[1][1], marker = '+')
#        plt.axis((math.floor(min(tot.min())),math.ceil(max(tot.max())),math.floor(min(tot.min())),math.ceil(max(tot.max()))))
        plt.axis((-140,30,-140,30))
        plt.title(title)
        plt.ylabel('Pelvis Accel')
        plt.xlabel('Chest Accel')
        plt.legend(loc = 4)
        plt.draw()
        plt.subplots_adjust(top = 0.96, bottom = 0.06, left = 0.09, right = 0.97, hspace = 0.34, wspace = 0.46)
        i = i + 1
#%% custom plot 2
#    plt.figure('C%d'% k)
#    i = 1
#    for TCN in TCNs:
#        chest = fulldata['Chest'][TCN]#.rolling(window=20,center=False).mean().shift(-10) 
#        pelvis = fulldata['Pelvis'][TCN]#.rolling(window=20,center=False).mean().shift(-10)
#        [g, c] = lookup_pop([TCN],'')
#        
#        points = [[[],[]],[[],[]]]
#        [t0, tm, st, mp] = get_peak(time, chest)
#        points[0][0].extend([t0,tm])
#        points[0][1].extend([st,mp])
#        
#        [t0, tm, st, mp] = get_peak(time, pelvis)
#        points[1][0].extend([t0,tm])
#        points[1][1].extend([st,mp])
#        
#        if points[0][0][1] < points[1][0][1]: #tch < tpel
#            tm = points[1][0][1]
#        else:
#            tm = points[0][0][1]
#            
#        chest = chest[time<tm]
#        pelvis = pelvis[time<tm]
##        pelvis = pelvis.append(pandas.Series([float('Nan') for i in range(len(pelvis),len(chest))]))
##        chest = chest.append(pandas.Series([float('Nan') for i in range(len(chest),len(pelvis))]))
#        pc = pelvis.subtract(chest).dropna()
#        
#        title = ', '.join(map(str, g.loc[0].tolist()[1:5]))
#        title = '\n'.join(wrap(title, 22))
#        plt.rcParams["axes.titlesize"] = 8
#        ylabel = 'Acceleration (X-direction) [g]'
#        
#        plt.subplot(3,3,i)
#        plt.plot(time[:len(pc)], pc, label = TCN)
#        plt.scatter(points[0][0], points[0][1], marker = '+')
#        plt.scatter(points[1][0], points[1][1], marker = '+')
#        plt.axis(bounds)
#        plt.title(title)
#        plt.ylabel('P-C Relative Accel')
#        plt.xlabel('Time')
#        plt.legend(loc = 4)
#        plt.draw()
#        plt.subplots_adjust(top = 0.96, bottom = 0.06, left = 0.09, right = 0.97, hspace = 0.34, wspace = 0.46)
#        i = i + 1    
    #%%
    k = k + 1
#%% integration
#chestint = np.append(scipy.integrate.cumtrapz(abs(chest), x = time, dx = 0.0001),[0])
#fig, ax1 = plt.subplots()
#ax1.plot(time,chest,'b')
#ax2 = ax1.twinx()
#ax2.plot(time,chestint,'g')