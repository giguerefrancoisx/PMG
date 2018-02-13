# -*- coding: utf-8 -*-
"""
MAIN FILE
    Runs the channel writing and plotting files

Created on Wed Oct 18 14:22:22 2017

@author: giguerf
"""
import os
import sys
if 'C:/Users/giguerf/Documents' not in sys.path:
    sys.path.insert(0, 'C:/Users/giguerf/Documents')
from GitHub.COM import writebook as wb
from SLED_PFW import plotbook

chlist = ['12CHST0000Y7ACXC', '12PELV0000Y7ACXA']
directory = os.fspath('P:/SLED/DATA/')

data = wb.writebook(chlist, directory)

for ch in chlist:
    data[ch].to_excel('P:/SLED/SAI/'+ch+'.xlsx', index = False)

plotbook()

print('All Done!')