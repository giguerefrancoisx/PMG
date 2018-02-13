# -*- coding: utf-8 -*-
"""
CRASH TYPE / SPEED GRIDS PLOT
    Compare the channel traces for each crash scenario

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

dummy = 'Q6'
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
figlist = ['A','B','C','D','E']
sublist = ['off48', 'mur48', 'off56', 'mur56']
linelist = ['14CHST0000'+dummy+'ACXC','14PELV0000'+dummy+'ACXA']

# Titles and subloc
figtitle = 'Relative Chest and Pelvis Accelerations:\n'
subtitle = ''

#subloc = dict(zip(sublist, [1, 4, 5, 4, 5, 2, 3, 1]))
subloc = dict([(subID, i + 1) for i, subID in enumerate(sublist)])
#bounds = [style.bounds(measure='AC') for i in range(4)]
#boundsdict = dict(zip(sublist, bounds))
sharex = 'all'
sharey = 'all'

# Data to use
time, fulldata, veh, sled = ob.openbook(readdir)
groupdict = ob.grids(fulldata, ['14CHST0000'+dummy+'ACXC','14PELV0000'+dummy+'ACXA'])
groupnames = {'A':'Clek','B':'Bubblebum','C':'Evenflo Evolve',
              'D':'Takata + Mifold','E':'All Others'}
titles = {'mur48': 'Type: Mur, Speed: 48 km/h',
          'mur56': 'Type: Mur, Speed: 56 km/h',
          'off48': 'Type: Offset, Speed: 48 km/h',
          'off56': 'Type: Offset, Speed: 56 km/h'}
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

for figID in figlist:  # groups
    datadict[figID] = {}
    linelabeldict[figID] = {}

    figtitledict[figID] = groupnames[figID]
    filename[figID] = 'Grid_'+groupnames[figID]

    for subID in sublist:  # grids
        datadict[figID][subID] = {}
        linelabeldict[figID][subID] = {}

        linecolordict[subID] = style.colordict(linelist)

#        pop = dict(zip(linelist, [genpop, cutpop]))
#        gencut = dict(zip(linelist, [gendict, cutdict]))

        subtitledict[subID] = titles[subID]

        for lineID in linelist:  # chest/pelvis
            try:
                datadict[figID][subID][lineID] = groupdict[figID][lineID][subID]
            except KeyError:
                print('Data not registered', figID, subID, lineID)
                datadict[figID][subID][lineID] = gplot.ignore(time)  # Make zero

            label = places[lineID]
            linelabeldict[figID][subID][lineID] = label

###---------------------------------------------------------------------------
# Combine arguments into a dictionary and pass to function in one word
# Do Not Modify
###---------------------------------------------------------------------------
gplot.plot(datadict, figlist, sublist, linelist, savedir, linelabeldict,
           linecolordict, filename, figtitle, subtitle, figtitledict, 
           subtitledict, subloc, sharex, sharey)
