# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:09:03 2018

@author: giguerf
"""

#%% COPY DTA FILES FROM AHEC TO THOR
#import pandas
#import shutil
#
#table = pandas.read_excel('P:/AHEC/ahectable.xlsx')
#table = table.dropna(axis=0, thresh=3).dropna(axis=1)
#table = table[table.loc[:,'CBL_11'].str.contains('TH')|table.loc[:,'BLR_11'].str.contains('TH')]
#cibles = table[table.loc[:,'CBL_11'].str.contains('TH')].CIBLE.tolist()
#beliers = table[table.loc[:,'BLR_11'].str.contains('TH')].BELIER.tolist()
#
#for subdir in ['Full Sample/48/','Full Sample/56/']:
#    for file in os.listdir(readdir+subdir):
#        if file[:-9] in cibles+beliers and file.endswith('.csv'):
#            print(file[:-9])
#            shutil.copyfile(readdir+subdir+file, readdir+'THOR/'+file)

#%% GET LABEL FROM PLOT ON CLICK
#fig = plt.figure()
##...
#ax = plt.gca()
#annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
#                    bbox=dict(boxstyle="round", fc="w"),
#                    arrowprops=dict(arrowstyle="->"))
#annot.set_visible(False)
#
#def on_plot_hover(event):
#    for curve in ax.get_lines():
#        if curve.contains(event)[0]:
#            #print "over %s" % curve.get_gid()
#            print(curve.get_label())
#            annot.set_text(curve.get_label())
#            annot.get_bbox_patch().set_alpha(0.4)
#
#fig.canvas.mpl_connect('button_press_event', on_plot_hover)
#plt.show()

#%% FIX OFFSET IN SPECIFIC CHANNELS
#import os
#import pandas
#
#directory = os.fspath('P:/AHEC/DATA/THOR/')
#channels = ['11CHSTLEUPTHDC0B','11CHSTRIUPTHDC0B','11CHSTLELOTHDC0B',
#            '11CHSTRILOTHDC0B','11CHSTLEUPTHDSXB','11CHSTRIUPTHDSXB',
#            '11CHSTLELOTHDSXB','11CHSTRILOTHDSXB']
#fulldata = {}
#for i, filename in enumerate(os.listdir(directory)):
#    if filename.endswith('.csv'):
#        test = pandas.read_csv(directory+filename)
#        test = test.drop('Unnamed: 0', axis=1)
#        chs = test.columns.intersection(channels)
#        test[chs] = test.loc[:,chs]-test.loc[100,chs]
#        test.to_csv(directory+filename, index=False)
#        print(i+1, ' of 79 completed')

#%% PRINT LIST OF EVERY CHANNEL FOUND IN AHEC TO GENERATE MASTER LIST
#chlist = []
#subdirs= ['Full Sample/48','Full Sample/56','HEV vs ICE/48','HEV vs ICE/56','OLD vs NEW/48','OLD vs NEW/56']
#import os
#import pandas
#from collections import Counter
#
#for subdir in subdirs:
#    print(subdir)
#    testframedict = {}
#    directory = os.fspath('P:/AHEC/DATA/')
#    count = len(os.listdir(directory+subdir))
#    i = 1
#    for filename in os.listdir(directory+subdir):
#        if filename.endswith(".xls"):
#            testframe = pandas.read_excel(directory+subdir+'/'+filename, sheetname = None, header = 0,index_col = 0,skiprows = [1,2])
#
#            if len(testframe) == 1:
#                testframe = list(testframe.items())[0][1]
#            else:
#                testframe = pandas.concat([list(testframe.items())[0][1],list(testframe.items())[1][1]], axis = 1)
#                ### Here NaN rows are inserted randomly into TC12-003, reason unknown
#                testframe = testframe.dropna(axis = 0) #temporary fix, investigate later
#
#            chlist.extend(testframe.columns.tolist())
#            testframedict[filename[:-9]] = testframe #trim 9 characters from end, which preserves '_2' in TCNs
#            per = i/count*100
#            i = i+1
#            print('%.1f %% Complete' % per)
#            continue
#        else:
#            continue
#
#chs = Counter(chlist)
#chset = [list(chs.keys())[i] for i in range(len(list(chs.keys()))) if list(chs.values())[i] >= 6] #arbitrary 6 repeats
#print(chset)
#%% Plot knee distance vs femur load
#from PMG.COM.table import get
#import matplotlib.pyplot as plt
#import scipy.stats
#import numpy as np
#table = get('THOR')
#table = table[table.CBL_BELT.isin(['SLIP','OK'])]
#table = table.drop(table[table.KNEES.isin(['NAN'])].index)
#slip = table[table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
#ok = table[table.CBL_BELT.isin(['OK'])].CIBLE.tolist()
#x_slip, y_slip = table[table.CIBLE.isin(slip)].KNEES, table[table.CIBLE.isin(slip)]['FEMUR MAX']
#x_ok, y_ok = table[table.CIBLE.isin(ok)].KNEES, table[table.CIBLE.isin(ok)]['FEMUR MAX']
#plt.figure()
#plt.plot(x_slip, y_slip, '.', label='slip')
#plt.plot(x_ok, y_ok, '.', label='ok')
#slope, intercept, r_value, *_ = scipy.stats.linregress(x_slip, y_slip)
#line_slip = np.poly1d((slope, intercept))
#plt.plot(x_slip, line_slip(x_slip), label='Slip, R^2 = {:.3f}'.format(r_value**2))
#slope, intercept, r_value, *_ = scipy.stats.linregress(x_ok, y_ok)
#line_slip = np.poly1d((slope, intercept))
#plt.plot(x_ok, line_slip(x_ok), label='Ok, R^2 = {:.3f}'.format(r_value**2))
#plt.legend()
