# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:30:52 2018
plotting functions
@author: tangk
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from plotly.offline import plot
import plotly.graph_objs as go
import seaborn as sns
from PMG.COM.helper import *
import plotly.tools as tls
import copy

def rename_legend(ax, names):
    """renames legend entries
    names is a dict of {oldname: newname}"""
    texts = ax.get_legend().get_texts()
    for t in texts:
        t.set_text(names[t.get_text()])


def initiate_missing_keys(d,keys,cls=None):
    d2 = copy.deepcopy(d)
    """initiates {key: None} that are not in dict d."""
    missing = [m for m in keys if m not in d2]
    for m in missing:
        d2[m] = copy.deepcopy(cls)
    return d2
        

def assign_missing_colours(d,k,line):
    """d is a dictionary of line specs. d[k] is a dictionary of line specs for 
    key k. If d[k] does not specify colour, then colour is assigned to the colour
    of line."""
    if d[k]==None:
        d[k] = {'color': line.get_color()}
    elif len(d[k])==0:
        d[k] = {'color': line.get_color()}
    elif 'color' not in d[k]: 
        d[k]['color'] = line.get_color()


def get_axes(num_subplots,hdim=6,vdim=4):
    """Creates figures with number of subplots defined by num_subplots
    Figure size is hardcoded and number of rows/columns are automatically 
    determined by get_figure_layout"""
    nrows, ncols = get_figure_layout(num_subplots)
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,sharey=True,figsize=(ncols*hdim, nrows*vdim))   
    return fig, axs


def get_figure_layout(num_subplots):
    """Returns the number of rows and columns for a figure with subplots 
    based on num_subplots"""
    if num_subplots==3:
        nrows = 1
        ncols = 3
    elif num_subplots in [7,8]:
        nrows = 4
    ncols = math.ceil(math.sqrt(num_subplots))
    nrows = math.ceil(num_subplots/ncols)
    return nrows,ncols
        

def get_order(order,d):
    """returns the order of plotting
    d is the dictionary"""
    if order==[]:
        return d.keys()
    else:
        return partial_ordered_intersect(order,d.keys())
    

def plot_categorical_scatter(ax,x,show_error=True,var=lambda x: pd.Series.std(pd.Series(x)),**kwargs):
    """plots a scatter plot with categories along the x axis and values along the y axis
    x: dict of {category name: samples} 
    var: error metric (e.g. std, variance). Default is pandas sample std.
    **kwargs are passed onto sns.stripplot. An important one is jitter."""
    labels = [k for k in x for i in range(len(x[k]))]
    values = [x[k][i] for k in x for i in range(len(x[k]))]
    
    if show_error:
        means = [np.mean(x[k]) for k in x]
        stds = [var(x[k]) for k in x]
        ax.errorbar(x.keys(), 
                    means, 
                    yerr=stds, 
                    ecolor='black', 
                    capsize=5, 
                    linestyle='none', 
                    marker='_', 
                    markersize=13, 
                    color='k')
    
    ax = sns.stripplot(x=labels, y=values, ax=ax, **kwargs)
    return ax


# to do: add plotting (normalized) densities
def plot_bs_distribution(ax,x,bins,ci={},c={}):
    """plots the boostrap distribution with confidence intervals
    x is a dict of {label: data}
    bins is a dict of {label: bins}
    c is a dict of {labels: face colours}
    ci is a dict of {labels: [lower_CI, upper_CI]"""
    c = initiate_missing_keys(c, x.keys())

    for k in x:
        patch = ax.hist(x[k], bins=bins[k], alpha=0.25, label=k, color=c[k])
        if c[k]==None:
            c[k] = patch[2][0].get_facecolor()[:3]
    
    if len(ci)>0:
        for k in ci:
            ax.hist(x[k][np.logical_or(x[k]<ci[k][0],x[k]>ci[k][1])], bins=bins[k], alpha=0.4, color=c[k])
    return ax


def get_indices(n_cat, df_size, width):
    """gets the indices for bar plots. n_cat is the number of categories (i.e. colours)
    df_size is the number of different conditions (i.e. x position)
    returns a list specifying the indices for each corresponding category"""
    indices_centers = n_cat*np.arange(df_size)
    if n_cat%2==0:
        # even number of categories
        indices = [indices_centers+i*width/2 for i in range(-n_cat//2,n_cat//2+1) if i!=0]
    else:
        # odd number of categories
        indices = [indices_centers+i*width for i in range(-n_cat//2+1,n_cat//2+1)]
    return indices, indices_centers


def plot_bar(ax,x,errorbar=True,var=pd.DataFrame.std,width=0.6,plot_specs={},order=[]):
    """plots a bar graph on ax. x is the input data in the form 
    {category name: DataFrame of (n_samples,n_classes)}. Bars are mean values
    across the samples. Error bar is optional and error metric (e.g. std, etc)
    is specified by var and is default the pandas std. all the dataframes in x
    should have the same number of columns. width is the bar width and plot_specs is 
    a dictionary of {category: {dict of plotting specs}}"""
    
    df_size = [x[k].shape[1] if len(x[k].shape)>1 else 1 for k in x]
    n_cat = len(x)
    plot_specs = initiate_missing_keys(plot_specs, x.keys(), cls={})
    
    if len(dict.fromkeys(df_size))>1:
        raise Exception('ncol should be the same for all DataFrames')
    
    indices, indices_centers = get_indices(n_cat, df_size[0], width)

    order = get_order(order, x)        
    for i, k in enumerate(order):
        mean = x[k].mean()
        if errorbar:
            err = var(x[k])
        else: 
            err = [np.nan for i in range(df_size[0])]
        ax.bar(indices[i],
               mean,
               width,
               yerr=err,
               label=k,
               capsize=6,
               error_kw={'elinewidth':2,'capthick':2},
               **plot_specs[k])
    ax.set_xticks(indices_centers)
    ax.set_xticklabels(x[k].columns)
    return ax


def add_stars(ax, x, p, y, orientation='v', **kwargs):
    """add stars (for significance) on axis ax at x at y*1.05 if p<0.05
    **kwargs goes into ax.text(). """
    if orientation=='v':
        for i in range(len(x)):
            if p[i]<0.001: 
                ax.text(x[i], 1.05*y[i], '***', horizontalalignment='center', **kwargs)
            elif p[i]<0.01:
                ax.text(x[i], 1.05*y[i], '**', horizontalalignment='center', **kwargs)
            elif p[i]<0.05:
                ax.text(x[i], 1.05*y[i], '*', horizontalalignment='center', **kwargs)
    elif orientation=='h':
        for i in range(len(x)):
            if p[i]<0.001: 
                ax.text(1.05*x[i], y[i], '***', va='center', **kwargs)
            elif p[i]<0.01:
                ax.text(1.05*x[i], y[i], '**', va='center', **kwargs)
            elif p[i]<0.05:
                ax.text(1.05*x[i], y[i], '*', va='center', **kwargs)
        
            
def plot_range(ax,x,c={},order=[]):
    """plots the range of values in x
    x is a dict of {labels: data}. for each label, the range is min(x[label]) to
    max(x[label]). c is a dict of colours."""
    c = initiate_missing_keys(c,x.keys())
    order = get_order(order, x)
    
    for i, k in enumerate(order):
        xmin = min(x[k])
        delta = max(x[k])-xmin
        ax.broken_barh([(xmin, delta)], (-2*i,1), color=c[k])
    ax.set_yticks(-2*np.asarray(range(len(x)))+0.5)
    ax.set_yticklabels(order)
    return ax


def plot_full(t,x,order=[]):
    """plots all elements of x in its own subplot
    x is a Series with labels in the index or a dict of {label: data} """
    fig, axs = get_axes(len(x))
    order = get_order(order, x)
    for i, k in enumerate(order):
        ax = axs.flatten()[i]
        ax.plot(t,x[k])
        ax.set_title(k)
    return fig, axs


def plot_ecdf(ax,x):
    """plots the ecdfs defined in x. x is a dict of {label name: ECDF derived from
    some scipy function??? Returns the axis"""
    for k in x:
        ax.plot(x[k].lowerlimit+np.linspace(0,x[k].binsize*x[k].cumcount.size,x[k].cumcount.size),
                x[k].cumcount,
                label=k)
    return ax

def plot_overlay(ax,t,x,line_specs={}):
    """plots overlays of line plots using data specified in x and time t onto ax
    x is a dict of {label name: pd.Series or nested array-like of data}
    line_specs is a dict of {label name: dict of associated line specs}"""
    line_specs = initiate_missing_keys(line_specs,x.keys(),cls={})

    for k in x:
        for i in x[k]:
            if np.isnan(i).all():
                continue
            if 'alpha' not in line_specs[k]:
                line = ax.plot(t, i, alpha=0.5, label=k, **line_specs[k])
            else:
                line = ax.plot(t, i, label=k, **line_specs[k])
            assign_missing_colours(line_specs,k,line[-1])
    return ax

def plot_overlay_2d(ax,x,y,line_specs={}):
    """plots overlay using data specified in x and y onto ax
    x and y are dicts of {label name: pd.Series of arrays specifiying x and y values}
    line_specs is a dict of {label name: dict of associated line specs}"""
    line_specs = initiate_missing_keys(line_specs,x.keys(),cls={})
    for k in x:
        for i, j in zip(*(x[k], y[k])):
            if np.isnan(i).all() or np.isnan(j).all():
                continue
            line = ax.plot(i, j, label=k, **line_specs[k])
            assign_missing_colours(line_specs,k,line[-1])
    return ax


def plot_bands(ax, t, x, **kwargs):
    """seaborn lineplot"""
    sns_df = []
    for k in x: 
        for i in x[k].index:
            df = pd.DataFrame({'time': t, 'signal': x[k][i]})
            df['grp'] = k
            sns_df.append(df)
    sns_df = pd.concat(sns_df, axis=0)
    ax = sns.lineplot(x='time', y='signal', hue='grp', data=sns_df, ax=ax, **kwargs)
    return ax

    
def plot_scatter(ax,x,y,marker_specs={}):
    """plots a scatter plot using data in x and y onto ax
    marker specs are dicts of {label name: dict of marker specs}"""
    marker_specs = initiate_missing_keys(marker_specs,x.keys(),cls={})
    # set default marker 
    for k in marker_specs:
        if marker_specs[k]==None:
            marker_specs[k] = {'marker': '.'}
        elif 'marker' not in marker_specs[k]:
            marker_specs[k]['marker'] = '.'
    
    for k in x:
        for i, j in zip(*(x[k], y[k])):
            if np.isnan(i).all() or np.isnan(j).all():
                continue
            line = ax.plot(i, j, label=k, linestyle='none', **marker_specs[k])
            assign_missing_colours(marker_specs,k,line[-1])
    return ax

        
def plot_scatter_with_labels(x, y, marker_specs={}):
    """creates a plotly graph with labels visible on hover.
    inputs are the same as plot_scatter.
    labels are determined from the index names in x.
    returns the plotly figure object."""
    
    labels = [l for k in x for l in x[k].index]
    fig, ax = plt.subplots()
    ax = plot_scatter(ax, x, y, marker_specs=marker_specs)
    plotly_fig = tls.mpl_to_plotly(fig)
    for i in range(len(plotly_fig['data'])):
        plotly_fig['data'][i]['name'] = labels[i]
        plotly_fig['data'][i]['hoverinfo'] = 'name'
    return plotly_fig
    

def get_legend_labels(ax):
    """gets the unique legend labels out of the lines/patches/whatever in ax.
    Returns a list of lines or bar containers"""
    legend_labels = {line.get_label(): line for line in ax.lines+ax.containers if line.get_label()!='_nolegend_' }
    return legend_labels


def set_labels(ax, labels):
    """sets the labels specified in dict label. all key: value pairs are
    label: labelname except legend, which is 'legend': kwargs. """
    if 'xlabel' in labels:
        ax.set_xlabel(labels['xlabel'])
    if 'ylabel' in labels:
        ax.set_ylabel(labels['ylabel'])
    if 'title' in labels:
        ax.set_title(labels['title'])
    if 'legend' in labels:
        # get the unique legend labels
        legend_labels = get_legend_labels(ax)
        ax.legend(handles=legend_labels.values(), **labels['legend'])
    return ax

def set_labels_plotly(fig, labels):
    """same as set_labels but input is a plotly figure and there is no legend
    specification."""
    if 'xlabel' in labels:
        fig['layout']['xaxis']['title'] = labels['xlabel']
    if 'ylabel' in labels:
        fig['layout']['yaxis']['title'] = labels['ylabel']
    if 'title' in labels:
        fig['layout']['title'] = labels['title']
    return fig
        

def adjust_font_sizes(ax, dictionary):
    """adjusts the font size of plot contained in ax
    dictionary specifies the changes, and should be in the format {'label': fontsize}
    possible keys are: 'ticklabels','xticklabel','yticklabel','legend','title','axlabels','xlabel','ylabel'"""
    
    if 'axlabels' in dictionary:
        ax.xaxis.get_label().set_fontsize(dictionary['axlabels'])
        ax.yaxis.get_label().set_fontsize(dictionary['axlabels'])
    
    if 'xlabel' in dictionary:
        ax.xaxis.get_label().set_fontsize(dictionary['xlabel'])
        
    if 'ylabel' in dictionary:
        ax.yaxis.get_label().set_fontsize(dictionary['ylabel'])
    
    if 'ticklabels' in dictionary:
        ax.tick_params(labelsize=dictionary['ticklabels'])
        
    if 'xticklabel' in dictionary:
        ax.tick_params(labelsize=dictionary['xticklabel'],axis='x')
    
    if 'yticklabel' in dictionary:
        ax.tick_params(labelsize=dictionary['yticklabel'],axis='y')
    
    if 'title' in dictionary:
        ax.title.set_fontsize(dictionary['title'])
        
    if 'legend' in dictionary:
        ax.legend_.get_texts()[0].set_fontsize(dictionary['legend'])        
    
    return ax


def do_anything(func, *args, **kwargs):
    """Literally do anything.
    func is the name of the function
    *args and **kwargs are arguments that are passed to func
    Returns whatever the function returns"""
    res = eval('{}(*args, **kwargs)'.format(func))
    return res