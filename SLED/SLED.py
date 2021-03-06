# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:53:00 2018

@author: giguerf
"""
from PMG.COM.writebook import writeHDF5
from PMG.COM.openbook import openHDF5
from PMG.COM import table as tb, data as dat

table = tb.get('SLED')

SLED = 'P:/SLED/Data/'
chlist = ['S0SLED000000ACXD','12HEAD0000Y7ACZA',
          '12HEAD0000Y7ACXA','12HEAD0000Y6ACXA','12HEAD0000Y2ACXA',
          '12CHST0000Y7ACXC','12CHST0000Y6ACXC','12CHST0000Y2ACXC',
          '12PELV0000Y7ACXA','12PELV0000Y6ACXA','12PELV0000Y2ACXA',
          '12CHST0000Y7ACRC', '12CHST0000Y7DSXB', '12HEAD0000Y7ACRA']

writeHDF5(SLED, chlist)

time, fulldata = openHDF5(SLED, chlist)

#time, clean_data = dat.import_data(SLED, chlist, check=True)
