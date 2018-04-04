# -*- coding: utf-8 -*-
"""
GROUPED PLOT
    Compare the channel means for the general and subset population

Created on Fri Nov 10 11:43:05 2017

@author: giguerf
"""
#import os
from PMG.COM.openbook import openHDF5
from PMG.COM import table as tb, plotstyle as style, globplot as gplot

table = tb.get('SLED')

SLED = 'P:/SLED/Data/'
chlist = ['12HEAD0000Y7ACXA','12HEAD0000Y2ACXA',
          '12CHST0000Y7ACXC','12CHST0000Y2ACXC',
          '12PELV0000Y7ACXA','12PELV0000Y2ACXA']

time, fulldata = openHDF5(SLED, chlist)

savedir = 'P:/SLED/Plots/gplot/'

#readdir = os.fspath('P:/BOOSTER/SAI/'+dummy)
#savedir = os.fspath('P:/BOOSTER/Plots/glob/') #'+dummy+'/')
#
#keys = [filename[:16] for filename in os.listdir(readdir)]
#values = ['Chest', 'Illiac Lower L', 'Illiac Upper L', 'Illiac Lower R',
#          'Illiac Upper R', 'Lumbar X', 'Lumbar Z', 'Pelvis']
#places = dict(zip(keys, values))

# %% GENERATE ARGUMENTS - for use in function call

###---------------------------------------------------------------------------
# Define main lists, titles, and subloc
# Define list dictionaries, other loop parameters
###---------------------------------------------------------------------------

# Main Lists
figlist = ['FigureID_0','FigureID_1','FigureID_2']
sublist = ['SubplotID_0','SubplotID_1']
linelist = ['LineID_0','LineID_1','LineID_2','LineID_3']

# Titles and subloc
figtitle = 'This Title is Common to All Figures'
subtitle = 'This subplot title is common to all subplots in a figure, usually blank'


# subloc: assign subplot number to each item in sublist. useful for
# skipping certain plots. Default is:
# subloc = dict(zip(sublist, range(1, len(sublist)+1)))
subloc = dict(zip(sublist, [1, 1]))

# blanks: assures you dont repeat colors if you plot twice to the same subplot
blanks = gplot.skipcolors(subloc, sublist)

# assign subplots to share axes. see help(style.subplots)
sharex = 'all'
sharey = 'all'

# specify xlimits for each subplot
xlim = dict(zip(sublist, [(0,0.2)]*len(sublist)))

###---------------------------------------------------------------------------
# Main Loop
###---------------------------------------------------------------------------

# for _ID in _list
#   some_dictionary = {} [leave alone]
#   ###
#   edit between markers
#   ###

# Initialize dictionaries
datadict = {}
figtitledict = {}
subtitledict = {}
linelabeldict = {}
linecolordict = {}
ylabeldict = {}
filename = {}
datadict['Time'] = time

for figID in figlist:
    datadict[figID] = {}
    linelabeldict[figID] = {}

    ###
    figtitledict[figID] = 'Figure title specific to each figure, appended to common title'
    filename[figID] = 'Filename specific to each figure'
    ###
    for subID in sublist:
        datadict[figID][subID] = {}
        linelabeldict[figID][subID] = {}

        ###
        keys = keys #specify keys for colors
        linecolordict[subID] = style.colordict(blanks[subID]+keys)
        subtitledict[subID] = 'subplot title specific to each subplot, appended to common title'
        ###

        for lineID in linelist:
            try:
                ###
                datadict[figID][subID][lineID] = get_data_for_line #some function to get data
                ###
            except KeyError:
                print('Data not registered', figID, subID, lineID)
#                    datadict[figID][subID][lineID] = None

            ###
            linelabeldict[figID][subID][lineID] = 'Label for each line'
            ###
###
show_stats = False # choose whether to show means + intervals. Applies to all figures.
plotargs = {'linewidth':1}
legendargs = {'loc':4}
###

###---------------------------------------------------------------------------
# Do Not Modify
###---------------------------------------------------------------------------
gplot.plot(datadict, figlist, sublist, linelist, linelabeldict, linecolordict,
           figtitle, subtitle, figtitledict, subtitledict, ylabeldict, subloc,
           sharex, sharey, savedir, filename,
           xlim, show_stats, plotargs, legendargs)
