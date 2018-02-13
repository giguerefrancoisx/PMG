# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:39:39 2017

@author: giguerf
"""

import os
import sys
sys.path.insert(0, 'C:/Users/giguerf/Documents/GitHub/COM/')
import writebook as wb
import matplotlib.pyplot as plt
import pandas
from collections import OrderedDict
#from lookup_pairs import lookup_pairs

def lookup_pairs(TCNs):

    pairs = pandas.DataFrame(TCNs,columns = ['OLD'])
    table = pandas.read_excel('P:/BOOSTER/boostertable.xlsx', index_col = 0)
    pairs['SEAT'] = table.loc[pairs['OLD'],'Modele'].tolist()
    pairs['GROUP'] = table.loc[pairs['OLD'],'Group'].tolist()
        
    return pairs

chlist = ['14CHST0000Y7ACXC', '14CHST0000Y7ACYC', '14CHST0000Y7ACZC', '14ILACLELOY7FOXB', '14ILACLEUPY7FOXB', '14LUSP0000Y7FOXA', '14LUSP0000Y7FOYA', '14LUSP0000Y7FOZA', '14LUSP0000Y7MOXA', '14LUSP0000Y7MOYA', '14PELV0000Y7ACXA', '14SPINUP00Y7ACXC', '16CHST0000Y7ACXC', '16CHST0000Y7ACYC', '16CHST0000Y7ACZC', '16ILACRILOY7FOXB', '16ILACRIUPY7FOXB', '16LUSP0000Y7FOXA', '16LUSP0000Y7FOYA', '16LUSP0000Y7FOZA', '16LUSP0000Y7MOXA', '16LUSP0000Y7MOYA', '16PELV0000Y7ACXA', '16PELV0000Y7ACYA', '16PELV0000Y7ACZA', '16SPINUP00Y7ACXC']
directory = os.fspath('P:/SLED/Prius Buck/')

data = wb.writebook(chlist, directory)

for chname in data:
    data[chname].columns = [str(col) + '_' + chname[:2] for col in data[chname].columns]
    data[chname].to_excel(directory+chname+'.xlsx', index = False)

#%%
readdir = directory
savedir = os.fspath('P:/SLED/Prius Buck/Plots/')
colors = {'Chest':'tab:blue','Pelvis':'tab:green','Chest2':'tab:orange','Pelvis2':'tab:purple'}
chdict = {}
TCNs = []
for filename in os.listdir(readdir):
    if filename.endswith('.xlsx'):
        
        chdata = pandas.read_excel(readdir+'/'+filename)
        chdict[filename[:16]] = chdata
        TCNs.extend(chdata.columns[1:].tolist())
        
    time = chdata.iloc[:,0] #renamed time channel for legibility
TCNs = list(set(TCNs))
TCNs.sort()
pairs = lookup_pairs(TCNs)

keys = list(set(pairs.SEAT))
values = ['tab:blue','tab:green','tab:orange','tab:purple','tab:pink']
colors = dict(zip(keys, values))
#%%
plt.close('all')
plt.figure('One', figsize=(20, 12.5))

plt.subplot(2,3,1)
[plt.plot(time, chdict['14ILACLELOY7FOXB'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = 'Lower '+pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['14ILACLELOY7FOXB'].columns[1:].tolist()]
[plt.plot(time, chdict['14ILACLEUPY7FOXB'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = 'Upper '+pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['14ILACLEUPY7FOXB'].columns[1:].tolist()]
[plt.plot(time, chdict['16ILACRILOY7FOXB'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = 'Lower '+pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['16ILACRILOY7FOXB'].columns[1:].tolist()]
[plt.plot(time, chdict['16ILACRIUPY7FOXB'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = 'Upper '+pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['16ILACRIUPY7FOXB'].columns[1:].tolist()]
plt.title('Upper & Lower Illiac Force')
plt.legend(loc = 5, fontsize = 'small')

plt.subplot(2,3,2)
[plt.plot(time, chdict['14ILACLEUPY7FOXB'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['14ILACLEUPY7FOXB'].columns[1:].tolist()]
[plt.plot(time, chdict['16ILACRIUPY7FOXB'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['16ILACRIUPY7FOXB'].columns[1:].tolist()]
plt.title('Upper Illiac Force')
plt.legend(loc = 5)

plt.subplot(2,3,3)
[plt.plot(time, chdict['14ILACLELOY7FOXB'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['14ILACLELOY7FOXB'].columns[1:].tolist()]
[plt.plot(time, chdict['16ILACRILOY7FOXB'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['16ILACRILOY7FOXB'].columns[1:].tolist()]
plt.title('Lower Illiac Force')
plt.legend(loc = 5)

plt.subplot(2,3,4)
[plt.plot(time, chdict['14SPINUP00Y7ACXC'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['14SPINUP00Y7ACXC'].columns[1:].tolist()]
[plt.plot(time, chdict['16SPINUP00Y7ACXC'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['16SPINUP00Y7ACXC'].columns[1:].tolist()]
plt.title('Upper Spine Acceleration')
plt.legend(loc = 5)

plt.subplot(2,3,5)
[plt.plot(time, chdict['14LUSP0000Y7FOXA'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['14LUSP0000Y7FOXA'].columns[1:].tolist()]
[plt.plot(time, chdict['14LUSP0000Y7FOZA'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['14LUSP0000Y7FOZA'].columns[1:].tolist()]
plt.title('Lumbar Spine Force, X')
plt.legend(loc = 5)

plt.subplot(2,3,6)
[plt.plot(time, chdict['14LUSP0000Y7FOZA'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['14LUSP0000Y7FOZA'].columns[1:].tolist()]
[plt.plot(time, chdict['16LUSP0000Y7FOZA'][tcn],color = colors[pairs[pairs.OLD == tcn].SEAT.tolist()[0]], label = pairs[pairs.OLD == tcn].SEAT.tolist()[0]) for tcn in chdict['16LUSP0000Y7FOZA'].columns[1:].tolist()]
plt.title('Lumbar Spine Force, Z')
plt.legend(loc = 5)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.xlim([0,0.2])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 1)

plt.tight_layout()
figManager = plt.get_current_fig_manager() # maximize window for visibility
figManager.window.showMaximized()
plt.savefig(savedir+'One.png', dpi = 200)
#%%
plt.figure('Two', figsize=(20, 12.5))


plt.subplot(2,2,1)
#    values = pandas.concat([chdict['14SPINUP00Y7ACXC'][tcn],chdict['14PELV0000Y7ACXA'].iloc[:,1:][tcn]], axis = 1)
[plt.plot(time, chdict['14SPINUP00Y7ACXC'][tcn], color = 'tab:blue', label = 'Spine') for tcn in ['TC14-503_2_14','TC14-503_4_14','TC14-503_5_14']]
[plt.plot(time, chdict['14PELV0000Y7ACXA'][tcn], color = 'tab:green', label = 'Pelvis') for tcn in ['TC14-503_2_14','TC14-503_4_14','TC14-503_5_14']]
plt.title('Upper Spine vs Pelvis Acceleration, Mifold')

plt.subplot(2,2,2)
#values = pandas.concat([chdict['14CHST0000Y7ACXC'][tcn],chdict['14PELV0000Y7ACXA'].iloc[:,1:][tcn]], axis = 1)
[plt.plot(time, chdict['14CHST0000Y7ACXC'][tcn], color = 'tab:blue', label = 'Chest') for tcn in ['TC14-503_2_14','TC14-503_4_14','TC14-503_5_14']]
[plt.plot(time, chdict['14PELV0000Y7ACXA'][tcn], color = 'tab:green', label = 'Pelvis') for tcn in ['TC14-503_2_14','TC14-503_4_14','TC14-503_5_14']]
#plt.plot(time, values, label = 'Data')
plt.title('Chest vs Pelvis Acceleration, Mifold')

plt.subplot(2,2,3)
#    values = pandas.concat([chdict['14SPINUP00Y7ACXC'][tcn],chdict['14PELV0000Y7ACXA'].iloc[:,1:][tcn]], axis = 1)
[plt.plot(time, chdict['14SPINUP00Y7ACXC'][tcn], color = 'tab:blue', label = 'Spine') for tcn in ['TC14-503_7_14','TC14-503_8_14','TC14-503_9_14']]
[plt.plot(time, chdict['14PELV0000Y7ACXA'][tcn], color = 'tab:green', label = 'Pelvis') for tcn in ['TC14-503_7_14','TC14-503_8_14','TC14-503_9_14']]
#plt.plot(time, values, label = 'Data')
plt.title('Upper Spine vs Pelvis Acceleration')

plt.subplot(2,2,4)
#    values = pandas.concat([chdict['14CHST0000Y7ACXC'][tcn],chdict['14PELV0000Y7ACXA'].iloc[:,1:][tcn]], axis = 1)
[plt.plot(time, chdict['14CHST0000Y7ACXC'][tcn], color = 'tab:blue', label = 'Chest') for tcn in ['TC14-503_7_14','TC14-503_8_14','TC14-503_9_14']]
[plt.plot(time, chdict['14PELV0000Y7ACXA'][tcn], color = 'tab:green', label = 'Pelvis') for tcn in ['TC14-503_7_14','TC14-503_8_14','TC14-503_9_14']]
#plt.plot(time, values, label = 'Data')
plt.title('Chest vs Pelvis Acceleration')

#plt.subplot(4,3,9)
#values = pandas.concat([chdict['14SPINUP00Y7ACXC'],chdict['16SPINUP00Y7ACXC'].iloc[:,1:]], axis = 1)
#plt.plot(time, values, label = 'Data')
#plt.title('Seat Belt Force at Shoulder')

plt.subplot(2,2,3)
#values = pandas.concat([chdict['16SPINUP00Y7ACXC'].iloc[:,1:],chdict['16PELV0000Y7ACXA'].iloc[:,1:]], axis = 1)
[plt.plot(time, chdict['16SPINUP00Y7ACXC'][tcn], color = 'tab:blue', label = 'Spine') for tcn in chdict['16SPINUP00Y7ACXC'].columns[1:].tolist()]
[plt.plot(time, chdict['16PELV0000Y7ACXA'][tcn], color = 'tab:green', label = 'Pelvis') for tcn in chdict['16PELV0000Y7ACXA'].columns[1:].tolist()]
#plt.plot(time, values, label = 'Data')
plt.title('Upper Spine vs Pelvis Acceleration')

plt.subplot(2,2,4)
#values = pandas.concat([chdict['16CHST0000Y7ACXC'].iloc[:,1:],chdict['16PELV0000Y7ACXA'].iloc[:,1:]], axis = 1)
#plt.plot(time, values, label = 'Data')
[plt.plot(time, chdict['16CHST0000Y7ACXC'][tcn], color = 'tab:blue', label = 'Chest') for tcn in chdict['16CHST0000Y7ACXC'].columns[1:].tolist()]
[plt.plot(time, chdict['16PELV0000Y7ACXA'][tcn], color = 'tab:green', label = 'Pelvis') for tcn in chdict['16PELV0000Y7ACXA'].columns[1:].tolist()]
plt.title('Chest vs Pelvis Acceleration')

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.xlim([0,0.2])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 4)

plt.tight_layout()
figManager = plt.get_current_fig_manager() # maximize window for visibility
figManager.window.showMaximized()
plt.savefig(savedir+'Two.png', dpi = 200)

#%% New legend labelling code
#plt.figure()
#lines = plt.plot(time, chdict['14CHST0000Y7ACXC'].iloc[:,1:].rolling(window=30,center=False).mean().shift(-15))
#lines2 = plt.plot(time, chdict['16CHST0000Y7ACXC'].iloc[:,1:])
#linelabels = chdict['14LUSP0000Y7FOXA'].columns[1:].tolist()+chdict['16LUSP0000Y7FOXA'].columns[1:].tolist()
#plt.legend(lines+lines2, linelabels)
#plt.figure()
#lines = plt.plot(time, chdict['14SPINUP00Y7ACXC'].iloc[:,1:])
#lines2 = plt.plot(time, chdict['16SPINUP00Y7ACXC'].iloc[:,1:])
#linelabels = chdict['14LUSP0000Y7FOXA'].columns[1:].tolist()+chdict['16LUSP0000Y7FOXA'].columns[1:].tolist()
#plt.legend(lines+lines2, linelabels)
#plt.figure()
#lines = plt.plot(time, chdict['14PELV0000Y7ACXA'].iloc[:,1:])
#lines2 = plt.plot(time, chdict['16PELV0000Y7ACXA'].iloc[:,1:])
#linelabels = chdict['14LUSP0000Y7FOXA'].columns[1:].tolist()+chdict['16LUSP0000Y7FOXA'].columns[1:].tolist()
#plt.legend(lines+lines2, linelabels)

#%% filtering with sav-gol
#import scipy.signal
##very noisy:
#plt.figure()
#line = chdict['14CHST0000Y7ACXC'].iloc[:,4]
#plt.plot(time,line, label = 'line')
#filtered = scipy.signal.savgol_filter(line, 31, 2)
#plt.plot(time,filtered, label = '31 - 2')
#filtered = scipy.signal.savgol_filter(line, 31, 5)
#plt.plot(time,filtered, label = '31 - 5')
#filtered = scipy.signal.savgol_filter(line, 91, 5)
#plt.plot(time,filtered, label = '91 - 5')
#plt.legend()
##less noisy:
#plt.figure()
#line = chdict['14CHST0000Y7ACXC'].iloc[:,5]
#plt.plot(time,line, label = 'line')
#filtered = scipy.signal.savgol_filter(line, 31, 2)
#plt.plot(time,filtered, label = '31 - 2')
#filtered = scipy.signal.savgol_filter(line, 31, 5)
#plt.plot(time,filtered, label = '31 - 5')
#filtered = scipy.signal.savgol_filter(line, 91, 5)
#plt.plot(time,filtered, label = '91 - 5')
#plt.legend()
