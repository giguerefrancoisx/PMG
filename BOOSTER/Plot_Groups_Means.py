# -*- coding: utf-8 -*-
"""
GROUPED PLOT
    Compare the channel means for the general and subset population

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
savedir = os.fspath('P:/BOOSTER/Plots/glob/') #'+dummy+'/')

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
figlist = ['ChestPelvis','UpperLowerIlliac','LumbarXZ']
sublist = ['General','Subset']
linelist = list(places.keys())

# Titles and subloc
figtitle = 'Comparision of the General and Subset Population Means:\n'
subtitle = ''

subloc = dict(zip(sublist, [1, 1]))
# subloc = dict([(subID, i + 1) for i, subID in enumerate(sublist)])
#bounds = [(None, None, None, None) for _ in sublist]
#boundsdict = dict(zip(sublist, bounds))
blanks = gplot.skipcolors(subloc, sublist)
sharex = 'all'
sharey = 'all'

channelpairs = [['14CHST0000Y7ACXC','14PELV0000Y7ACXA'],
                ['14ILACLELOY7FOXB','14ILACLEUPY7FOXB'],
                ['14LUSP0000Y7FOXA','14LUSP0000Y7FOZA']]
channelpairs = dict(zip(figlist, channelpairs))

# Data to use
time, gendict, cutdict, genpop, cutpop = ob.gencut(readdir, 'D')
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

for figID in figlist:  # channel pairs
    datadict[figID] = {}
    linelabeldict[figID] = {}

    titlechannels = [places[ch] for ch in keys if ch in channelpairs[figID]]
    figtitledict[figID] = ', '.join(map(str,titlechannels))
    filename[figID] = 'Groups_Means_'+figID

    for subID in sublist:  # gen/cut
        datadict[figID][subID] = {}
        linelabeldict[figID][subID] = {}

        linecolordict[subID] = style.colordict(blanks[subID]+keys)

#        pop = dict(zip(linelist, [genpop, cutpop]))
        gencut = dict(zip(sublist, [gendict, cutdict]))

        subtitledict[subID] = ''

        for lineID in linelist:  # channels
            if lineID in channelpairs[figID]:
                try:
                    datadict[figID][subID][lineID] = gencut[subID][lineID+'_stats']['Mean']
                except KeyError:
                    print('Data not registered', figID, subID, lineID)
                    datadict[figID][subID][lineID] = gplot.ignore(time)  # Make zero
            else:
                datadict[figID][subID][lineID] = None

            label = places[lineID]+', '+subID
            linelabeldict[figID][subID][lineID] = label

###---------------------------------------------------------------------------
# Combine arguments into a dictionary and pass to function in one word
# Do Not Modify
###---------------------------------------------------------------------------
gplot.plot(datadict, figlist, sublist, linelist, savedir, linelabeldict,
           linecolordict, filename, figtitle, subtitle, figtitledict, 
           subtitledict, subloc, sharex, sharey)
