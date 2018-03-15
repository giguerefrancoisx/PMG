# -*- coding: utf-8 -*-
"""
PLOT STYLE FUNCTIONS
    Makes easily readable commands for common plotting functions
Created on Tue Nov  7 16:13:57 2017

@author: gigue
"""
import numpy as np
import pandas as pd
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
from PMG.COM.data import clean_outliers

def colordict(data, by='order', values=None):
    """Creates a dictionary of color assignments based on an iterable or
    DataFrame.

    Input
    -----------
    data : iterable or DataFrame
        keys will be derived from this object. For a DataFrame or Series, column labels
        are used. For iterable, index is used
    by : 'max, 'min', or 'order'
        for an iterable, max and min will sort the list by it's values in
        ascending and descending order, respectively. For a DataFrame, 'max'
        and 'min' will order the list of columns by the maximum and minimum
        value of each column. 'order' will preserve original order for all
        types.

    values : any sequence whose items are recognised by matplotlib as a color
        If None, default color cycle is used. Listed and Linear segmented
        colormaps can be used, as well as iterables will color IDs as name
        strings, hex strings, rgb tuples, etc. See color.py for list of
        colormaps in matplotlib.

    Returns
    -----------
    colors : dict
        a dictionary whose keys are the list passed to this function and whose
        values are colors in a valid matplotlib representation.

    Examples
    -----------
    >>> colors = colordict(['TC11-239', 'TC14-214', 'TC14-220', 'TC17-211', 'TC17-206'], by='order', values=plt.cm.viridis)
    >>> colors
    >>> {'TC11-239': [0.267004, 0.004874, 0.329415],
         'TC14-214': [0.253935, 0.265254, 0.529983],
         'TC14-220': [0.163625, 0.471133, 0.558148],
         'TC17-206': [0.477504, 0.821444, 0.318195],
         'TC17-211': [0.134692, 0.658636, 0.517649]}
    """
    if isinstance(values, clrs.LinearSegmentedColormap):
        N = values.N-1
    else:
        N=1

    if isinstance(data, (pd.DataFrame, pd.Series)):
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data).T
        if by == 'max': #TODO Fix me to use 0-1 min to max and just reverse colormap
            maxlist = 1-(data.max()-data.max().min())/(data.max().max()-data.max().min())
#            maxlist = (data.max().max()-data.max())/(data.max().max()-data.max().min())
            keys = maxlist.sort_values().index
            maxlist[np.isnan(maxlist)] = 0
            mapping = (maxlist[keys]*N).round(0).astype(int)
        elif by == 'min':
            minlist = (data.min()-data.min().min())/(data.min().max()-data.min().min())
            keys = minlist.sort_values().index
            minlist[np.isnan(minlist)] = 0
            mapping = (minlist[keys]*N).round(0).astype(int)
        elif by == 'mean':
            scale = 1-(data.mean()-data.mean().min())/(data.mean().max()-data.mean().min())
            keys = scale.sort_values().index
            scale[np.isnan(scale)] = 0
            mapping = (scale[keys]*N).round(0).astype(int)
        elif by == 'order':
            keys = data.columns
            mapping = np.arange(len(data.columns))*N//(len(data.columns)-1)
        else:
            raise Exception('No other coloring methods implemented yet')
    else:
        if by == 'max':
            maxlist = (data-min(data))/(max(data)-min(data))
            keys = np.flip(np.argsort(data), 0)
            mapping = (maxlist[keys]*N).round(0).astype(int)
        elif by == 'min':
            minlist = (data-min(data))/(max(data)-min(data))
            keys = np.argsort(data)
            mapping = (minlist[keys]*N).round(0).astype(int)
        elif by == 'order':
            keys = data
            mapping = np.arange(len(data))*N//(len(data)-1)
        else:
            raise Exception('No other coloring methods implemented yet')

    if values is None:
        values = plt.rcParamsDefault['axes.prop_cycle'].by_key()['color']

    if isinstance(values, clrs.ListedColormap):
        cmap = values
        n_steps = len(cmap.colors)//len(keys)
        if n_steps >= 1:
            values = cmap.colors[::n_steps]
        else:
            values = cmap.colors
        #values = camp.colors[mapping]

    elif isinstance(values, clrs.LinearSegmentedColormap):
        cmap = values
#        n_steps = cmap.N//len(keys)
#        if n_steps >= 1:
#            values = [cmap(n)[:3] for n in range(0, cmap.N, n_steps)]
#        else:
#            values = [cmap(n)[:3] for n in range(cmap.N)]
        values = [rgb[:3] for rgb in cmap(mapping)]

    if len(keys) > len(values):
        values = [values[k%len(values)] for k in range(len(keys))]
    colordict = dict(zip(keys, values))

    return colordict

def labels(ax, title, ylabel):
    """Sets the axis limits and labels"""

    ax.set_title(title)
    ax.set_xlim((0, 0.3))
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time [s]')

def explode(d, dv):
    """Explodes nested dictionaries into a list of entries"""

    for k, v in list(d.items()):
        if isinstance(v, dict):
            explode(v, dv)
        else:
            dv.append(v)

    return dv

def ylim_no_outliers(data, scale=1.08):
    """Returns the ylimits necessary to correctly plot the dataset passed as
    input. You may pass a dataframe or list of dataframes to evaluate.
    """

    clean = clean_outliers(data, stage=1)
    ymin, ymax = clean.min().min(), clean.max().max()

    ymax = scale*ymax if ymax > ymin/10 else ymin/10
    ymin = scale*ymin if ymin < -ymax/10 else -ymax/10

    return ymin, ymax

def ylabel(dimension, direction):
    """Returns ylabel"""

    if dimension == 'AC':
        ylabel = 'Acceleration ('+direction+'-direction) [g]'
    elif dimension == 'FO':
        if direction in ['X','Y','Z']:
            ylabel = 'Force ('+direction+'-direction) [N]'
        else:
            ylabel = 'Force [N]'
    elif (dimension == 'DC') or (dimension == 'DS'):
        if direction in ['X','Y','Z']:
            ylabel = 'Displacement ('+direction+'-direction) [mm]'
        else:
            ylabel = 'Displacement [mm]'
    elif dimension == 'AV':
        ylabel = 'Angle Velocity ('+direction+'-direction) [deg/sec]'
    else:
        ylabel = 'Type not recognised'

    return ylabel

def legend(ax, loc, **kwargs):
    """Removes duplicate line labels and creates legend at desired location"""

    from collections import OrderedDict
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    lgd = ax.legend(by_label.values(), by_label.keys(), loc = loc, **kwargs)
    return lgd

def maximize():
    """Maximizes the plot window"""
    figManager = plt.get_current_fig_manager() # maximize window for visibility
    figManager.window.showMaximized()
    plt.show()
    fig = plt.gcf()
    fig.tight_layout()

def save(savedir, filename, dpi=100):
    """Saves the current figure in the chosen directory"""

    plt.subplots_adjust(top=0.93, bottom=0.05, left=0.06, right=0.96,
                        hspace=0.23, wspace=0.20)
    plt.savefig(savedir+filename, dpi = dpi)

def isinvalid(share):
    """Determines if the list is a valid assignment"""

    if any([share[i]>i+1 for i in range(len(share))]): #loc greater than position
        return True
    elif any([share[i]!=share[share[i]-1] for i in range(len(share))]):
        #referenced location doesnt match position, already sharing. (does this matter?)
        return True
    else:
        return False

def sqfactors(n, axratio=1, figratio=1.6):
    """
    Returns the squarest grid arrangement r x c for n items
    figratio from figuresize (20,12.5)
    axratio is the minimum ratio (ex: >1, wider than tall)
    """
    if n == 0:
        return 0,0
    if axratio>figratio:
        raise ValueError('Illegal parameter configuration')

    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i== 0:
            factors.extend([i, n//i])
    factors.sort()
    m = len(factors)//2
    factors = factors[m-1:m+1] #take middle factors
    if (factors[1]/factors[0] > figratio/axratio) and n != 2:
        factors = sqfactors(n+1, axratio=axratio, figratio=figratio)

    return factors

def subplots(r, c, sharex='none', sharey='none', visible=True, figsize=None, **kwargs):
    """
    Input
        r, c: number of rows and columns in subplot grid
        sharex: string or list indicating how to share the x axis

    sharex and sharey can be passed as a list of subplot locations sharing
    the x or y axis, respectively.
    Example: sharey = [1,2,3,4] will create individual y axes.
             sharey = [1,2,1,4] will share the y axis for plot 1 and 3
             sharey = [1,1,1,1] will share the y axes for all plots
    sharex and sharey can also be assigned 'all', 'none', 'row', or 'col'.
    'row' will share the specified axis along the rows, for example.

    Returns
        fig, axs
    """
    if figsize is None:
        figsize = (5*c, 3.125*r)
#    axs = [] #Why is this here?

    if any([sharex == 'all', sharex == 'row', sharex == 'col',
            sharex == 'none']) and any([sharey == 'all', sharey == 'row',
                                        sharey == 'col', sharey == 'none']):
        fig, axs = plt.subplots(r,c, sharex=sharex, sharey=sharey,
                                squeeze=False, figsize=figsize, **kwargs)
        axs = axs.reshape(r*c)

        if visible:
            for ax in axs.flatten():
                for tk in ax.get_yticklabels():
                    tk.set_visible(True)
                for tk in ax.get_xticklabels():
                    tk.set_visible(True)

                ax.xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
                ax.xaxis.offsetText.set_visible(True)
                ax.yaxis.set_tick_params(which='both', labelleft=True, labelright=False)
                ax.yaxis.offsetText.set_visible(True)

        return fig, axs

    elif any([sharex == 'all', sharex == 'row', sharex == 'col',
              sharex == 'none']):
        arrays = {'all':[1 for i in range(len(sharey))],
                  'row':[(i//c)*c+1 for i in range(len(sharey))],
                  'col':[i%c+1 for i in range(len(sharey))],
                  'none':[i+1 for i in range(len(sharey))]}
        sharex = arrays[sharex]

    elif any([sharey == 'all', sharey == 'row', sharey == 'col',
              sharey == 'none']):
        arrays = {'all':[1 for i in range(len(sharex))],
                  'row':[(i//c)*c+1 for i in range(len(sharex))],
                  'col':[i%c+1 for i in range(len(sharex))],
                  'none':[i+1 for i in range(len(sharex))]}
        sharey = arrays[sharey]

    if all([type(sharex) == list, type(sharey) == list]):

        if len(sharex) != len(sharey):
            print('Warning! sharex and sharey lists are unequal length. The longer list will be truncated.')

        elif isinvalid(sharex) or isinvalid(sharey):
            print('You have misassigned the shared axis locations. Please revise.\n')
            confirm = input('Show Errors?\n')

            if confirm in ['yes','ok']:
                print('x:', sharex, '\ny:', sharey)
                print('x (loc, value): ', [(i+1, sharex[i]) for i in range(len(sharex)) if sharex[i]>i+1])
                print('y (loc, value): ', [(i+1, sharey[i]) for i in range(len(sharey)) if sharey[i]>i+1])
                print('x (loc, value, assigned): ', [(i+1, sharex[i], sharex[sharex[i]-1]) for i in range(len(sharex)) if sharex[i]!=sharex[sharex[i]-1]])
                print('y (loc, value, assigned): ', [(i+1, sharey[i], sharey[sharey[i]-1]) for i in range(len(sharey)) if sharey[i]!=sharey[sharey[i]-1]])
        else:
            fig = plt.figure(figsize=figsize, **kwargs) #num = 'title', figsize = (x,y)
            for i, (x, y) in enumerate(zip(sharex,sharey)):

                shx = (None if x-1 == i else axs[x-1])
                shy = (None if y-1 == i else axs[y-1])

                ax = plt.subplot(r, c, i+1, sharex = shx, sharey = shy)
                axs.append(ax)
    else:
        raise TypeError('Selection not possible. Check sharex and sharey')
        axs = None

    return fig, axs
