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