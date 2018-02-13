# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:10:07 2018

@author: tangk
"""
import scipy.cluster.hierarchy as hier
from scipy.spatial.distance import squareform, pdist
import numpy as np

#%%
def do_hcluster(linkMe,metric = ['euclidean','cityblock','correlation'],method = ['single','complete','average','centroid','median','ward']):        

    # linkMe: m x n DataFrame; m: sample no. n: time point
    # metric: metrics to try
    # method: methods to try
    
    writeMe = []
    # do linkage
    coph = {}
    inconsist = {}
    links = {}
    for mtc in metric:
        for mtd in method:
            if (mtd=='centroid' or mtd=='median' or mtd=='ward') and not(mtc=='euclidean'):
                continue
            
            # write to file
            writeMe.append('\n\n---------------------------------------------------------------------------------------------------------------------\n' + 
                           mtc + ' ' + mtd + '\n---------------------------------------------------------------------------------------------------------------------\n')
            
            if not(mtc in links):
                links[mtc] = {mtd:hier.linkage(linkMe,method=mtd,metric=mtc,optimal_ordering=True)}
                coph[mtc] = {mtd: hier.cophenet(links[mtc][mtd],pdist(linkMe))}
                inconsist[mtc] = {mtd: hier.inconsistent(links[mtc][mtd])}
            else:
                links[mtc].update({mtd:hier.linkage(linkMe,method=mtd,metric=mtc,optimal_ordering=True)})
                coph[mtc].update({mtd: hier.cophenet(links[mtc][mtd],pdist(linkMe))})
                inconsist[mtc].update({mtd: hier.inconsistent(links[mtc][mtd])})
            
            # append diagnostics
            writeMe.append('Inconsistency Matrix\n')
            writeMe.append(np.array2string(inconsist[mtc][mtd]))
            writeMe.append('\n\nCophenetic distance\n')
            writeMe.append(np.array2string(squareform(coph[mtc][mtd][1])))
            writeMe.append('\n\nCophenetic Correlation Coefficient: ')
            writeMe.append(np.array2string(coph[mtc][mtd][0]))
    return links, coph, inconsist, writeMe

def sax_hcluster(linkMe,method=['single','complete','average']):
    # linkMe is a compressed vector of minimum distances
    
    writeMe = []
    coph = {}
    inconsist = {}
    links = {}
    for mtd in method:
        writeMe.append('\n\n---------------------------------------------------------------------------------------------------------------------\n' + 
                       mtd + '\n---------------------------------------------------------------------------------------------------------------------\n')
        links[mtd] = hier.linkage(linkMe,method=mtd,optimal_ordering=True)
        coph[mtd]  = hier.cophenet(links[mtd],linkMe)
        inconsist[mtd] = hier.inconsistent(links[mtd])
        
        # append diagnostics
        writeMe.append('Inconsistency Matrix\n')
        writeMe.append(np.array2string(inconsist[mtd]))
        writeMe.append('\n\nCophenetic distance\n')
        writeMe.append(np.array2string(squareform(coph[mtd][1])))
        writeMe.append('\n\nCophenetic Correlation Coefficient: ')
        writeMe.append(np.array2string(coph[mtd][0]))
        
    return links, coph, inconsist, writeMe
