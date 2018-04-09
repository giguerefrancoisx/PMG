# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:30:52 2018

@author: tangk
"""
# plotting functions

import scipy.cluster.hierarchy as hier
import matplotlib.pyplot as plt
import numpy as np

# plot dendrogram
def plot_dendrogram(links,models,mtc,mtd):
    f = plt.figure(figsize=(20,10))
    if mtc==[]:
        plt.title('Method: ' + mtd)
        hier.dendrogram(links[mtd],leaf_rotation=90.,leaf_font_size=10.,labels=models)
    else:
        plt.title('Metric: ' + mtc + '; Method: ' + mtd)
        hier.dendrogram(links[mtc][mtd],leaf_rotation=90.,leaf_font_size=10.,labels=models)
    plt.xlabel('Model')
    plt.ylabel('Distance')
    return f
    
# plot full dataset    
def plot_full(t,linkMe,models,subplot_dim,fig_size):
    fig, axs = plt.subplots(subplot_dim[0],subplot_dim[1], sharey = 'all',figsize=fig_size)
    
    for i, ax in enumerate(axs.flatten()):
        if i>=len(linkMe):
            break
        data = linkMe.iloc[i,:]
        ax.plot(t,data)
        ax.set_title(models[i])
        ax.set_ylabel('Acceleration [g]')
        ax.set_xlim(0,0.15)
        ax.set_xlabel('Time [s]')
    return fig

# to do: consolidate this with the above function
def plot_full_2(t,data1,data2):
    files1 = data1.index
    n1 = int(np.ceil(len(files1)/10))
    
    if len(data2)>0:
        files2 = data2.index
        n2 = int(np.ceil(len(files2)/10))
    else:
        n2=0
        
    fig, axs = plt.subplots(n1+n2,10,sharey='all',figsize=(40,4*(n1+n2)))
    for i, ax in enumerate(axs[:n1].flatten()[range(len(files1))]):
        if len(data1[i]==1) and np.isnan(data1[i]).all():
            continue
        else:
            ax.plot(t,data1[i])
            ax.set_title(files1[i])
    if len(data2)>0:
        for i, ax in enumerate(axs[n1:].flatten()[range(len(files2))]):
            if len(data2[i]==1) and np.isnan(data2[i]).all():
                continue
            else:
                ax.plot(t,data2[i])
                ax.set_title(files2[i])    

# bar plot
# to do: expand to plot variable number of inputs
def plot_bar(ax, data):
    catlist = list(data.keys())
    sample_mean = []
    sample_std = []
    for cat in catlist:
        sample_mean.append(np.mean(data[cat]))
        sample_std.append(np.std(data[cat]))
    ax.bar(catlist,sample_mean)
    for cat in catlist:
        ax.plot([cat],[data[cat]],'.',markersize=10)
    ax.errorbar(catlist,sample_mean,yerr=sample_std,ecolor='black',capsize=5,linestyle='none')
    return ax

def plot_cat_nobar(ax,data):
    catlist = list(data.keys())
    sample_mean = []
    sample_std = []
    maxcat = 0
    for cat in catlist:
        sample_mean.append(np.mean(data[cat]))
        sample_std.append(np.std(data[cat]))
        ax.plot([cat],[data[cat]],'.',markersize=10)
        if len(data[cat])>maxcat:
            maxcat = len(data[cat])
    if maxcat > 1:
        ax.errorbar(catlist,sample_mean,yerr=sample_std,ecolor='black',capsize=5,linestyle='none',marker='_',markersize=13,color='k')
    return ax

# ecdf plot
# to do: expand to plot variable number of inputs
def plot_ecdf(ax,labels,c1,c2):    
    ax.plot(c1.lowerlimit+np.linspace(0,c1.binsize*c1.cumcount.size,c1.cumcount.size),c1.cumcount,label=labels[0])
    ax.plot(c2.lowerlimit+np.linspace(0,c2.binsize*c2.cumcount.size,c2.cumcount.size),c2.cumcount,label=labels[1])
    ax.legend()
    return ax

# plot at least one line plot. then use eval to plot others
def plot_multiple(ax,plotin):
    for i in range(len(plotin)):
        if len(plotin[i])==1:
            ax.plot(plotin[i])
        elif len(plotin[i])==2:
            ax.plot(plotin[i][0],plotin[i][1])
        elif len(plotin[i])==3:
            eval('ax.plot(plotin[i][0],plotin[i][1],' + plotin[i][2] + ')')
    return ax

def plot_bs_distribution(ax,x1,x2,ci=None):
    if len(x1)<3 or len(x2)<3:
        print('Not enough inputs!')
        return
    
    ax.hist(x1[0],bins=x1[1],alpha=0.25,label=x1[2],color='b')
    ax.hist(x2[0],bins=x2[1],alpha=0.25,label=x2[2],color='r')
    if not(ci==None):
        plt.hist(x1[0][np.logical_and(x1[0]>ci[0][0], x1[0]<ci[0][1])],bins=x1[1],alpha=0.4,color='b')
        plt.hist(x2[0][np.logical_and(x2[0]>ci[1][0], x2[0]<ci[1][1])],bins=x2[1],alpha=0.4,color='r')
    ax.axvline(np.mean(x1[0]),color='k',label=x1[2] + 'Mean')
    ax.axvline(np.mean(x2[0]),color='k',label=x2[2] + 'Mean')
    ax.legend()
    return ax

def plot_overlay(ax,t,x1,x2):
    for i in range(len(x1[0])):
        if np.isnan(x1[0][i]).all():
            continue
        if len(x1[0])==1:
            ax.plot(t,x1[0],color='b',alpha=0.5,label=x1[1])
        else:
            ax.plot(t,x1[0][i],color='b',alpha=0.5,label=x1[1])
    for i in range(len(x2[0])):
        if np.isnan(x2[0][i]).all():
            continue
        if len(x2[0])==1:
            ax.plot(t,x2[0],color='r',alpha=0.5,label=x2[1])
        else:
            ax.plot(t,x2[0][i],color='r',alpha=0.5,label=x2[1])
    return ax
    