# -*- coding: utf-8 -*-
"""
    GLOBAL PLOTTING FUNCTION
        A system of nested loops to auto-generate plots from lists and data

Created on Thu Nov  9 13:55:39 2017

@author: giguerf
"""
#%% BELOW IS AN EXAMPLE USAGE OF THE SET-UP CODE FOLLOWED BY THE MAIN FUNCTION
###----------------------------------------------------------------------------
#from PMG.COM.openbook import openHDF5
#from PMG.COM import table as tb, plotstyle as style, globplot as gplot
#
#table = tb.get('PROJECT_NAME')
#
#PATH = 'Path to Data Storage'
#chlist = ['Channel_0','Channel_1']
#
#time, fulldata = openHDF5(PATH, chlist)
#
#savedir = 'Path to Figures'
#
## Define objects used later on, if necessary
#
#
## %% GENERATE ARGUMENTS - for use in function call
#
####---------------------------------------------------------------------------
## Define main lists, titles, and subloc
## Define list dictionaries, other loop parameters
####---------------------------------------------------------------------------
#
## Main Lists
#figlist = ['FigureID_0','FigureID_1','FigureID_2']
#sublist = ['SubplotID_0','SubplotID_1']
#linelist = ['LineID_0','LineID_1','LineID_2','LineID_3']
#
## Titles and subloc
#figtitle = 'This Title is Common to All Figures'
#subtitle = 'This subplot title is common to all subplots in a figure, usually blank'
#
#
## subloc: assign subplot number to each item in sublist. useful for
## skipping certain plots. Default is:
## subloc = dict(zip(sublist, range(1, len(sublist)+1)))
#subloc = dict(zip(sublist, [1, 1]))
#
## blanks: assures you dont repeat colors if you plot twice to the same subplot
#blanks = gplot.skipcolors(subloc, sublist)
#
## assign subplots to share axes. see help(style.subplots)
#sharex = 'all'
#sharey = 'all'
#
## specify xlimits for each subplot
#xlim = dict(zip(sublist, [(0,0.2)]*len(sublist)))
#
####---------------------------------------------------------------------------
## Main Loop
####---------------------------------------------------------------------------
#
## for _ID in _list
##   some_dictionary = {} [leave alone]
##   ###
##   edit between markers
##   ###
#
## Initialize dictionaries
#datadict = {}
#figtitledict = {}
#subtitledict = {}
#linelabeldict = {}
#linecolordict = {}
#ylabeldict = {}
#filename = {}
#datadict['Time'] = time
#
#for figID in figlist:
#    datadict[figID] = {}
#    linelabeldict[figID] = {}
#
#    ###
#    figtitledict[figID] = 'Figure title specific to each figure, appended to common title'
#    filename[figID] = 'Filename specific to each figure'
#    ylabeldict = dict(zip(sublist, ['Highback','Highback','Lowback','Lowback']))
#    ###
#    for subID in sublist:
#        datadict[figID][subID] = {}
#        linelabeldict[figID][subID] = {}
#
#        ###
#        keys = keys #specify keys for colors
#        linecolordict[subID] = style.colordict(blanks[subID]+keys)
#        subtitledict[subID] = 'subplot title specific to each subplot, appended to common title'
#        ###
#
#        for lineID in linelist:
#            try:
#                ###
#                datadict[figID][subID][lineID] = get_data_for_line #some function to get data
#                ###
#            except KeyError:
#                print('Data not registered', figID, subID, lineID)
##                    datadict[figID][subID][lineID] = None
#
#            ###
#            linelabeldict[figID][subID][lineID] = 'Label for each line'
#            ###
####
#show_stats = False # choose whether to show means + intervals. Applies to all figures.
#plotargs = {'linewidth':1}
#legendargs = {'loc':4}
####
#
####---------------------------------------------------------------------------
## Do Not Modify
####---------------------------------------------------------------------------
#gplot.plot(datadict, figlist, sublist, linelist, linelabeldict, linecolordict,
#           figtitle, subtitle, figtitledict, subtitledict, ylabeldict, subloc,
#           sharex, sharey, savedir, filename,
#           xlim, show_stats, plotargs, legendargs)

#%% MAIN FUNCTION - No need to change for each application

###---------------------------------------------------------------------------
# DO NOT MODIFY BELOW THIS
###---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from PMG.COM import plotstyle as style

def skipcolors(subloc, linelist):
    """Determines the offset for line colors"""

    from collections import Counter
    subvals = list(subloc.values())
    inst = [0]*len(subvals)
    for i, loc in enumerate(subvals):
        inst[i] = inst[i] + Counter(subvals[:i])[loc]
    offset = [['' for _ in range(i*len(linelist))]+linelist for i in inst]
    offsetlist = dict(zip(list(subloc.keys()), offset))

    return offsetlist

def plot(datadict, figlist, sublist, linelist, linelabeldict, linecolordict,
         figtitle, subtitle, figtitledict, subtitledict, ylabeldict, subloc,
         sharex, sharey, savedir, filename,
         xlim, show_stats=False, plotargs={}, legendargs={}):
    """Uses the pre-built data and styling functions to output plots"""
#if 1 == 1: #for debugging as a script

    plt.close('all')
    time = datadict['Time']

    for figID in figlist:

        r, c = style.sqfactors(len(set(subloc.values())))
        fig, axs = style.subplots(r, c, sharex, sharey, num=figID, figsize=(10*c, 6.25*r))
        plt.suptitle(figtitle+figtitledict[figID])

        for subID in sublist:

            ax = axs[subloc[subID]-1]

            for lineID in linelist:

                color = linecolordict[subID][lineID]
                label = linelabeldict[figID][subID][lineID]
                lines = datadict[figID][subID][lineID]

                try:

                    if show_stats:
                        lines = lines.mean(axis=1)
                        alpha=0.05
                        low = lines.quantile(alpha, axis=1)
                        high = lines.quantile(alpha, axis=1)
                        ax.fill_between(time, low, high, color=color, label=label+' 90th', alpha=0.5)
                    ax.plot(time, lines, color=color, label=label, **plotargs)

                    _, *N = lines.shape
                    annotation = 1 if N==[] else N[0]

                except KeyError:

                    print('Unable to plot: ', figID, subID, lineID)
                    annotation = 0

            ax.plot(np.nan, np.nan, label='n = {}'.format(annotation), alpha=0, **plotargs)

            ax.set_title(subtitle+subtitledict[subID])
            ax.set_ylabel(ylabeldict[subID])
            ax.set_xlabel('Time [s]')
            ax.set_xlim(*xlim[subID])

            style.legend(ax, **legendargs)#, loc=4)

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
        plt.savefig(savedir+filename[figID]+'.png', dpi=200)
        plt.close('all')

### Must carefully construct argument items to avoid exception

#%% Notes, extra code
