# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:59:31 2018

@author: tangk
"""

from PMG.COM.openbook import openHDF5
from PMG.COM import arrange
from PMG.COM.get_props import *
from PMG.COM.plotfuns import *
from PMG.COM.data import import_data
import pandas as pd
from scipy.stats import anderson_ksamp
from scipy.stats import cumfreq
from PMG.read_data import read_merged
from PMG.COM import table as tb

dummy = 'Y6'
plotfigs = 1
savefigs = 1
writefiles = 1
usesmth = 0

#%%
channels = ['12CHST0000Y6DSXB',
            '12CHST0000Y6DSXA',
            '12HEAD0000Y6ACRA',
            '12CHST0000Y6ACRC',
            '12PELV0000Y6ACRA',
            '12HEAD0000Y6ACXA',
            '12CHST0000Y6ACXC',
            '12PELV0000Y6ACXA']
wherepeaks = np.array(['-tive','-tive','+tive','+tive','+tive','-tive','-tive','+tive'])
cutoff = range(100,1600)

table_y7 = table.query('DUMMY==\'' + dummy + '\'').filter(items=['SE','MODEL','SLED'])
table_y7 = table_y7.set_index('SE',drop=True)
if not(exclude==[]):
    table_y7 = table_y7.drop(exclude,axis=0)
models = np.unique(table_y7['MODEL'])
sleds = np.unique(table_y7['SLED'])

t, fulldata = import_data(directory,channels,tcns=table_y7.index)
chdata = arrange.test_ch_from_chdict(fulldata,cutoff)
t = t.get_values()[cutoff]
writename = 'C:\\Users\\tangk\\Python\\Sled_' + dummy + '_'