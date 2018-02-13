# -*- coding: utf-8 -*-
"""
NEW TEST COMPARISON PLOT
    Compare the channel traces from position 14 and 16 for tests recently added

Created on Fri Nov 10 11:43:05 2017

@author: giguerf
"""
import os
import sys
if 'C:/Users/giguerf/Documents' not in sys.path:
    sys.path.insert(0, 'C:/Users/giguerf/Documents')
from GitHub.COM import openbook as ob
from GitHub.COM import plotstyle as style
from GitHub.COM import globplot as gplot

dummy = 'Y7'
readdir = os.fspath('P:/BOOSTER/SAI/'+dummy)
savedir = os.fspath('P:/BOOSTER/Plots/'+dummy+'/')

keys = [filename[:16] for filename in os.listdir(readdir)]
values = ['Chest', 'Illiac Lower L', 'Illiac Upper L', 'Illiac Lower R',
          'Illiac Upper R', 'Lumbar X', 'Lumbar Z', 'Pelvis']
places = dict(zip(keys, values))

# %% GENERATE ARGUMENTS - for use in function call

###---------------------------------------------------------------------------
# Define main lists, titles, and subloc
# Define list dictionaries, other loop parameters
###---------------------------------------------------------------------------

# Main Lists
#figlist = ['TC18-105','TC12-004','TC16-127','TC16-129','TC16-132','TC17-205','TC17-207'] ##Deal with some being Q6
figlist = ['TC14-503_7']
sublist = list(places.keys())
linelist = ['14', '16']

# Titles and subloc
figtitle = 'Plot Comparison for Newest Tests:\n'
subtitle = ''

subloc = dict(zip(sublist, [1, 4, 5, 4, 5, 2, 3, 1]))
# subloc = dict([(subID, i + 1) for i, subID in enumerate(sublist)])
offsetlist = gplot.skipcolors(subloc, linelist)

sharex = 'all'
sharey = [1,2,2,4,4]

# Data to use
time, gendict, cutdict, genpop, cutpop = ob.gencut(readdir, group='')
#%%
###---------------------------------------------------------------------------
# Notes on Usage and main loop
###---------------------------------------------------------------------------

# for _ID in _list
#   datadict = {} [leave alone]
#   [shorthands go here]
#   vars = [change this]

datadict = {}
figtitledict = {}
subtitledict = {}
linelabeldict = {}
linecolordict = {}
filename = {}
datadict['Time'] = time

for figID in figlist:  # TCNs
    datadict[figID] = {}
    linelabeldict[figID] = {}

    speed = str(genpop[genpop['ALL'] == figID + '_14'].Vitesse.tolist()[0]) + ' km/h'

    figtitledict[figID] = figID + ' at ' + speed
    filename[figID] = figID

    for subID in sublist:  # places
        datadict[figID][subID] = {}
        linelabeldict[figID][subID] = {}

        linecolordict[subID] = style.colordict(offsetlist[subID])

        subtitledict[subID] = places[subID]

        for lineID in linelist:  # pos
            
            try:
                datadict[figID][subID][lineID] = gendict[subID][figID + '_' + lineID]
            except KeyError:
                print('Data not registered', figID, subID, lineID)
                datadict[figID][subID][lineID] = gplot.ignore(time)  # Make zero

            try:
                marque = genpop[genpop['ALL'] == figID + '_' + lineID].Marque.tolist()[0]
                label = places[subID] + ': Pos. ' + lineID + ': ' + marque
                linelabeldict[figID][subID][lineID] = label
            except IndexError:
                linelabeldict[figID][subID][lineID] = ''

###---------------------------------------------------------------------------
# Combine arguments into a dictionary and pass to function in one word
# Do Not Modify
###---------------------------------------------------------------------------
gplot.plot(datadict, figlist, sublist, linelist, savedir, linelabeldict,
           linecolordict, filename, figtitle, subtitle, figtitledict, 
           subtitledict, subloc, sharex, sharey)
