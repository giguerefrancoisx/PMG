# -*- coding: utf-8 -*-
"""
PLOT STYLE FUNCTIONS
    Makes easily readable commands for common plotting functions
Created on Tue Nov  7 16:13:57 2017

@author: gigue
"""
### Code to import this file
#from GitHub.COM.plotstyle import ''
import pandas as pd
import matplotlib.pyplot as plt
from PMG.COM.data import clean_outliers

def colordict(keys):
    """Assigns a tab color to each item in the list"""

    values = ['tab:blue', 'tab:green','tab:orange', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive','tab:cyan']
    if len(keys) > len(values):
        values = [values[k%len(values)] for k in range(len(keys))]
    colors = dict(zip(keys, values))

    return colors

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

def ylim(data):
    """
    Returns ylim matching argument.
    Pass 'data' as a dict with series objects
    or 'measure' as a string eg. 'AC', 'FO'
    """

    import math
#    ymax = max([abs(i) for i in [ymin,ymax]])
#    ymin = -ymax
#    bounds = (0, 0.2, ymin, ymax)

    ### FROM DATA
    if isinstance(data, dict):
        dv = explode(data, [])
        try:
            df = pd.concat(dv, axis = 1)
        except TypeError as err:
#            print('Dictionary contains non-DataFrame objects at root')
            raise TypeError('Dictionary contains non-DataFrame objects at depth')
#            x = input('Show list?\n')
#            if x in ['yes', 'ok']:
#                print(dv)
#                return dv
        ymin = 1.08*min(df.min())
        ymax = 1.08*max(df.max())
        ylim = (ymin, ymax) #add floor, ceil?

        if any([math.isnan(lim) for lim in ylim]):
            ylim = (None, None)
    else:
        raise TypeError('Data must be a dictionary of DataFrames')

    return ylim

def ylim_no_outliers(data, scale=1.08):
    """Returns the ylimits necessary to correctly plot the dataset passed as
    input. You may pass a dataframe or list of dataframes to evaluate.
    """

    ymin, ymax = clean_outliers(data, 'limits')

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

def legend(ax, loc):
    """Removes duplicate line labels and creates legend at desired location"""

    from collections import OrderedDict
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    lgd = ax.legend(by_label.values(), by_label.keys(), loc = loc)
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
