# -*- coding: utf-8 -*-
"""
PLOT THOR SAMPLE BY MODEL
    Compare the channel traces for instances of a model

Created on Fri Nov 10 11:43:05 2017

@author: giguerf
"""
import os
from GitHub.COM import openbook as ob
from GitHub.COM import plotstyle as style
from GitHub.COM import globplot as gplot

readdir = os.fspath('P:/AHEC/SAI/')
savedir = os.fspath('P:/AHEC/Plots/THOR/')

# %% GENERATE ARGUMENTS - for use in function call

###---------------------------------------------------------------------------
# Define main lists, titles, and subloc
# Define list dictionaries, other loop parameters
###---------------------------------------------------------------------------

# Main Lists
figlist = ['One']
sublist = ['CAMRY', 'CHEROKEE', 'CRUZE', 'VOLT', 'ESCAPE', 'ESCAPE HYBRID', 'FOCUS',
           'JETTA', 'MAZDA', 'PACIFICA', 'PRIUS', 'ROGUE', 'SONIC', 'TAURUS',
            'TIGUAN', 'YARIS']#['Camry','Cruze','Jetta','Escape','Mazda','Civic','Focus','Rogue']
linelist = ['10CVEHCG0000ACXD']

# Titles and subloc
figtitle = 'Vehicle CG accelerations by model'
subtitle = ''

#subloc = dict(zip(sublist, [1, 4, 5, 4, 5, 2, 3, 1]))
subloc = dict([(subID, i + 1) for i, subID in enumerate(sublist)])
sharex = 'all'
sharey = 'all'

# Data to use

time, fulldata = ob.openbook('P:/AHEC/SAI/')
singles, pairs = ob.lookup_pairs(project='THOR')
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

    figtitledict[figID] = ''
    filename[figID] = 'Figure_'+figID

    for subID in sublist:  # grids
        datadict[figID][subID] = {}
        linelabeldict[figID][subID] = {}

        linecolordict[subID] = style.colordict(linelist)

        subtitledict[subID] = subID
        car = singles[singles['CBL_MODELE'].str.contains(subID.upper())]

        for lineID in linelist:  # chest/pelvis
            try:
                datadict[figID][subID][lineID] = fulldata[lineID].loc[:,car.CIBLE]
            except KeyError:
                print('Data not registered', figID, subID, lineID)
                datadict[figID][subID][lineID] = gplot.ignore(time)  # Make zero

            label = 'CG-x'
            linelabeldict[figID][subID][lineID] = label

#%%---------------------------------------------------------------------------
# Combine arguments into a dictionary and pass to function in one word
# Do Not Modify
###---------------------------------------------------------------------------
gplot.plot(datadict, figlist, sublist, linelist, savedir, linelabeldict,
           linecolordict, filename, figtitle, subtitle, figtitledict,
           subtitledict, subloc, sharex, sharey)

