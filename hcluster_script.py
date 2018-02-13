# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:19:13 2018

@author: tangk
"""
# script for hierarchical clustering

import read_data, os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import numpy as np
import hcluster
from importlib import reload
import plotfuns
import saxpy


#%% parameters
dir1 = 'P:\\AHEC\\Data\\OLD vs NEW\\48\\'
dir2 = 'P:\\AHEC\\Data\\OLD vs NEW\\56\\'
#dir2 = []
cutoff = 1600
savename_append_at_top = 'OLD_vs_NEW_ALL_'
diagfile_name = savename_append_at_top + 'diagnostic.txt'
plotfigs = 1
savefigs = 1
writediag = 1
dosax = 0
readdata = 1
if dosax:
    alphabet_size = 7
    nwindow = 160
ch = 'VEHCG00' # vehicle CG
d = 'ACX' # vehicle CG
chname = '10CVEHCG0000ACXD'

#ch = '11PELV0000TH' #THOR pelvis
#d = 'ACX' #THOR pelvis
#chname = '11PELV0000THACXx'
    
#ch = '11NECKLO00TH' # THOR lower neck 
#d = 'FOX' # force in x
#chname = '11NECKLO00THFOXA'
    
    
    
#delrange = range(900,1100)

# HEV vs ICE 56, Old vs New 48
#subplot_dim = [2,4]
#fig_size = (30,10)

# HEV vs ICE 48
subplot_dim = [4,4]
fig_size = (40,30)

# Old vs New 56
#subplot_dim = [2,3]
#fig_size = (40,20)

# Full Sample 48
#subplot_dim = [6,7]
#fig_size = (60,35)

# Full Sample 56
#subplot_dim = [4,7]
#fig_size = (60,20) 

# HEV vs ICE all
#subplot_dim = [4,6]
#fig_size = (60,30)

#%% read data

#pctrl = ['TC17-025','TC15-035','TC14-220','TC15-035','TC15-162','TC17-028']

if readdata:
    files = []
    for filename in os.listdir(dir1):
        if filename=='TC11-007(SAI).csv' or filename=='TC12-013(SAI).csv':
            continue
        if filename.endswith('.csv'):
#        if filename.endswith('.csv') and filename[:-9] in pctrl:
            files.append(filename[:-9])        
    full_data = read_data.read_merged(dir1,files)
    n48files = len(files)
    
    if not(dir2==[]):
        for filename in os.listdir(dir2):
            if filename=='TC11-007(SAI).csv' or filename=='TC12-013(SAI).csv':
                continue
            if filename.endswith('.csv'):
    #        if filename.endswith('.csv') and filename[:-9] in pctrl:
                files.append(filename[:-9])
        
        full_data.update(read_data.read_merged(dir2,files[n48files:]))

    
#%% rearrange for linkage
CG_data = {}
xdata = {}
for test in full_data: # get channel information as dict
    CG_data[test] = full_data[test].filter(like=ch)
    xtest = CG_data[test].filter(like=d).get_values().flatten()[:cutoff]
    if len(xtest)>0:
        xdata[test] = xtest
        #cutting 0.04-0.06s out
#        xdata[test] = np.delete(xdata[test],delrange)

 
# linkMe is a n x m DataFrame, where n is the number of samples
linkMe = pd.DataFrame.from_dict(xdata).transpose()

if dosax:
    s = saxpy.SAX(wordSize=nwindow,alphabetSize=alphabet_size)
    saxrep = []
    mindist = []
    
    # for each pair
    for i in range(len(linkMe)):
        # get sax representation
        saxrep.append(s.to_letter_rep(s.normalize(linkMe.iloc[i,:]))[0])
    
    # compute minimum distance
    for i in range(len(saxrep)):
        for j in range(i+1,len(saxrep)):
            d = s.compare_strings(saxrep[i],saxrep[j])
            mindist.append(d)
#            if d==0:
#                raise Exception('0 distance!')
#            else:
#                mindist.append(d)
    
    
    # get linkage matrix
    links, coph, inconsist, writeMe = hcluster.sax_hcluster(mindist)
else:
    # call function to get linkage and diagnostics
    links, coph, inconsist, writeMe = hcluster.do_hcluster(linkMe)

if writediag:
    diagfile = open(diagfile_name,'w')
    writeMe.insert(0,savename_append_at_top + 'DIAGNOSTIC')
    diagfile.writelines(writeMe)                
    diagfile.close()

#%% get corresponding model names
models = []
table = pd.read_excel('P:/AHEC/ahectable.xlsx')
for f in list(xdata.keys()):
    m = table[table['CIBLE']==f]['CBL_MODELE'].get_values()
    if len(m)==0:
        m = table[table['BELIER']==f]['BLR_MODELE'].get_values()
        if len(m)==0:
            m = ' '
        else:
            m = m.item()
    else:
        m = m.item()
    m = f + ' (' + m+ ')'
    models.append(m)

models.sort()
#%% plot dendrogram
if plotfigs:
    if dosax:
        method = list(links.keys())
        
        for mtd in method:
            f = plotfuns.plot_dendrogram(links,models,[],mtd)
            if savefigs:
                f.savefig(savename_append_at_top + mtd + '.png',bbox_inches='tight')
    else:
        metric = list(links.keys())
        method = []
        
        for mtc in metric:
            method= method + list(links[mtc].keys())
        method = list(set(method))
        
        for mtc in metric:
            for mtd in method:
                if (mtd=='centroid' or mtd=='median' or mtd=='ward') and not(mtc=='euclidean'):
                    continue
                f = plotfuns.plot_dendrogram(links,models,mtc,mtd)
                if savefigs:
                    f.savefig(savename_append_at_top + mtc + '_' + mtd + '.png',bbox_inches='tight')

#%% plot full dataset

t = full_data[files[0]].iloc[:cutoff,0].get_values().flatten()
#t = np.delete(t,delrange)   
# save full plots
if savefigs:
    f = plotfuns.plot_full(t,linkMe,models,subplot_dim,fig_size)
    if savefigs:
        f.savefig(savename_append_at_top + chname + '.png',bbox_inches='tight')
        
