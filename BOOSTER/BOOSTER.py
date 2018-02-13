# -*- coding: utf-8 -*-
"""
MAIN FILE
    Runs the channel writing and plotting files

Created on Wed Oct 18 14:22:22 2017

@author: giguerf
"""
import os
import pandas
from GitHub.COM.writebook import writebook
#from BOOSTER.BOOSTER_PFW import plotbook

#if 'C:/Users/giguerf/Documents' not in sys.path:
#    sys.path.insert(0, 'C:/Users/giguerf/Documents')
#from GitHub.COM import writebook as wb
#from BOOSTER_PFW import plotbook

dummy = 'Y7'
readdir = os.fspath('P:/BOOSTER/DATA/')
subdir = os.fspath(dummy+'/')

if dummy == 'Y7':
    chlist14 = ['14CHST0000Y7ACXC', '14PELV0000Y7ACXA', '14LUSP0000Y7FOXA', '14LUSP0000Y7FOZA', '14ILACLELOY7FOXB', '14ILACLEUPY7FOXB','14ILACRILOY7FOXB', '14ILACRIUPY7FOXB']
    chlist16 = ['16CHST0000Y7ACXC', '16PELV0000Y7ACXA', '16LUSP0000Y7FOXA', '16LUSP0000Y7FOZA', '16ILACRILOY7FOXB', '16ILACRIUPY7FOXB','16ILACLELOY7FOXB', '16ILACLEUPY7FOXB']
    chlist = chlist14 + chlist16
elif dummy == 'Q6':
    chlist14 = ['14CHST0000Q6ACXC', '14PELV0000Q6ACXA', '14LUSP0000Q6FOXA', '14LUSP0000Q6FOZA', '14ILACLELOQ6FOXB', '14ILACLEUPQ6FOXB','14ILACRILOQ6FOXB', '14ILACRIUPQ6FOXB']
    chlist16 = ['16CHST0000Q6ACXC', '16PELV0000Q6ACXA', '16LUSP0000Q6FOXA', '16LUSP0000Q6FOZA', '16ILACRILOQ6FOXB', '16ILACRIUPQ6FOXB','16ILACLELOQ6FOXB', '16ILACLEUPQ6FOXB']
    chlist = chlist14 + chlist16

data = writebook(chlist, readdir+subdir)

for chname in data:
    data[chname].columns = ['_'.join([col, chname[:2]]) for col in data[chname].columns]

for ch14, ch16 in zip(chlist14, chlist16):
    export = pandas.concat([data[ch14], data[ch16].iloc[:,1:]], axis = 1)
    export.to_csv('P:/BOOSTER/SAI/'+subdir+ch14+'.csv', index = False)

#plotbook(subdir)

print('All Done!')