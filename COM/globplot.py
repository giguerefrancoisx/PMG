# -*- coding: utf-8 -*-
"""
    GLOBAL PLOTTING FUNCTION
        A system of nested loops to auto-generate plots from lists and data

Created on Thu Nov  9 13:55:39 2017

@author: giguerf
"""
#%% BELOW IS AN EXAMPLE USAGE OF THE SET-UP CODE FOLLOWED BY THE MAIN FUNCTION
###----------------------------------------------------------------------------
##%% CASE SET-UP - Attempt to recreate 'new test' figures
#
#import os
#import sys
#if 'C:/Users/giguerf/Documents' not in sys.path:
#    sys.path.insert(0, 'C:/Users/giguerf/Documents')
#from GitHub.COM import openbook as ob
#from GitHub.COM import plotstyle as style
#from GitHub.COM import globplot as gplot
#
#readdir = os.fspath('P:/BOOSTER/SAI')
#savedir = os.fspath('P:/BOOSTER/Plots/glob/')
#
#keys = [filename[:16] for filename in os.listdir(readdir)]
#values = ['Chest', 'Illiac_LowerL', 'Illiac_UpperL', 'Illiac_LowerR',
#          'Illiac_UpperR', 'Lumbar_X', 'Lumbar_Z', 'Pelvis']
#places = dict(zip(keys, values))
#
## %% GENERATE ARGUMENTS - for use in function call
#
####---------------------------------------------------------------------------
## Define main lists, titles, and subloc
## Define list dictionaries, other loop parameters
####---------------------------------------------------------------------------
#
## Main Lists
#figlist = ['TC16-127', 'TC12-004', 'TC17-205','TC16-129','TC18-105']
#sublist = list(places.keys())
#linelist = ['14', '16']
#
## Titles and subloc
#figtitle = 'Plot Comparison for Newest Tests:\n'
#subtitle = ''
#
#subloc = dict(zip(sublist, [1, 4, 5, 4, 5, 2, 3, 1]))
## subloc = dict([(subID, i + 1) for i, subID in enumerate(sublist)])
#blanks = gplot.skipcolors(subloc, linelist)
#
#sharex = 'all'
#sharey = 'all'
#
## Data to use
#time, gendict, cutdict, genpop, cutpop = ob.gencut(readdir, '')
##%%
####---------------------------------------------------------------------------
## Notes on Usage and main loop
####---------------------------------------------------------------------------
#
## for _ID in _list
##   datadict = {} [leave alone]
##   [shorthands go here]
##   vars = [change this]
#
#datadict = {}
#figtitledict = {}
#subtitledict = {}
#linelabeldict = {}
#linecolordict = {}
#filename = {}
#datadict['Time'] = time
#
#for figID in figlist:  # TCNs
#    datadict[figID] = {}
#    linelabeldict[figID] = {}
#
#    speed = str(genpop[genpop['ALL'] == figID + '_14'].Vitesse.tolist()[0]) + ' km/h'
#
#    figtitledict[figID] = figID + ' at ' + speed
#    filename[figID] = figID
#
#    for subID in sublist:  # places
#        datadict[figID][subID] = {}
#        linelabeldict[figID][subID] = {}
#
#        linecolordict[subID] = style.colordict(blanks[subID] + linelist)
#
#        subtitledict[subID] = places[subID]
#
#        for lineID in linelist:  # pos
#
#            try:
#                datadict[figID][subID][lineID] = gendict[subID][figID + '_' + lineID]
#            except KeyError:
#                print('Data not registered', figID, subID, lineID)
#                datadict[figID][subID][lineID] = gplot.ignore(time)  # Make zero
#
#            marque = genpop[genpop['ALL'] == figID + '_' + lineID].Marque.tolist()[0]
#            label = places[subID] + ': Pos. ' + lineID + ': ' + marque
#
#            linelabeldict[figID][subID][lineID] = label
#
####---------------------------------------------------------------------------
## Combine arguments into a dictionary and pass to function in one word
## Do Not Modify
####---------------------------------------------------------------------------
#gplot.plot(datadict, figlist, sublist, linelist, savedir, linelabeldict,
#           linecolordict, filename, figtitle, subtitle, figtitledict,
#           subtitledict, subloc, sharex, sharey)

#%% MAIN FUNCTION - No need to change for each application

###---------------------------------------------------------------------------
# DO NOT MODIFY BELOW THIS
###---------------------------------------------------------------------------
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

def ignore(time):
    """Returns a NaN series to plot silently"""
    zero = time - time
    return zero.replace(to_replace = 0, value = float('nan'))

def plot(datadict, figlist, sublist, linelist, savedir, linelabeldict,
         linecolordict, filename, figtitle, subtitle, figtitledict,
         subtitledict, subloc, sharex, sharey):
    """Uses the pre-built data and styling functions to output plots"""
#if 1 == 1: #for debugging as a script

    import matplotlib.pyplot as plt
    from GitHub.COM import plotstyle as style
    from GitHub.COM import trace_data as trace

    plt.close('all')
    time = datadict['Time']
    align_data = False
    show_means = False

    for figID in figlist:

        r, c = style.sqfactors(len(set(subloc.values())))
        fig, axs = style.subplots(r, c, sharex=sharex, sharey=sharey, visible=True, num=figID, figsize=(10*c, 6.25*r))
        plt.suptitle(figtitle+figtitledict[figID])

        subplot_data = []
        subplot_data = style.explode(datadict[figID], subplot_data)

        for subID in sublist:

            ax = axs[subloc[subID]-1]
            title = subtitle+subtitledict[subID]

#            channel = subID         ##Define channel for subID here

            for lineID in linelist:

                color = linecolordict[subID][lineID]
                label = linelabeldict[figID][subID][lineID]

                lines = datadict[figID][subID][lineID]

                try:

                    if align_data:
                        for line in lines:
                            time_line = time - trace.peak(time, lines[line])['t0'] + 0.05
                            ax.plot(time_line, lines[line], color=color, label=label)
                    else:
                        if show_means:
                            lines = lines.mean(axis=1)

                        ax.plot(time, lines, '.', color=color, label=label, markersize=0.5)

                    if len(lines.shape)>1:
                        annotation = lines.shape[1]
                    else:
                        annotation = 1

                except KeyError:

                    print('Unable to plot: ', figID, subID, lineID)
                    annotation = 0

            ax.plot(time, ignore(time), '.', label='n = {}'.format(annotation),  markersize=0)
            ax.set_ylim(style.ylim_no_outliers(subplot_data))
#            ylabel = style.ylabel(channel[12:14], channel[14:15])
            ylabel = 'Acceleration (X-Direction) [g]'
            style.labels(ax, title, ylabel)
#            ax.annotate('n = %d' % annotation, (0.01, 0.01), xycoords='axes fraction')
            #insert annotation as legend entry? append to label?
            style.legend(ax, loc=4)

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
        plt.savefig(savedir+filename[figID]+'.png', dpi=200)
        plt.close('all')

### Must carefully construct argument items to avoid exception

#%% Notes, extra code
