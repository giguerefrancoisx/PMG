# -*- coding: utf-8 -*-
"""
    GLOBAL PLOTTING FUNCTION
        A system of nested loops to auto-generate plots from lists and data

Created on Thu Nov  9 13:55:39 2017

@author: giguerf
"""
#%% BELOW IS AN EXAMPLE USAGE OF THE SET-UP CODE FOLLOWED BY THE MAIN FUNCTION
###----------------------------------------------------------------------------


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
         xlim=(0,0.3), show_stats=False, plotargs={}, legendargs={}):
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

            ax.plot(np.nan, np.nan, label='n = {}'.format(annotation), **plotargs)

            ax.set_title(subtitle+subtitledict[subID])
            ax.set_ylabel(ylabeldict[subID])
            ax.set_xlabel('Time [s]')
            ax.set_xlim(*xlim)

            style.legend(ax, **legendargs)#, loc=4)

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
        plt.savefig(savedir+filename[figID]+'.png', dpi=200)
        plt.close('all')

### Must carefully construct argument items to avoid exception

#%% Notes, extra code
