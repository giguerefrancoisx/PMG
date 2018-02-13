# -*- coding: utf-8 -*-
"""
MAIN FILE
    Runs the channel writing and plotting files for the chosen subdirectories

Created on Wed Oct 18 14:22:22 2017

@author: giguerf
"""
import os
from PMG.COM.writebook import writeHDF5
from PMG.COM.channels import codestring
from PMG.THOR.Plot_THOR import plotbook as plot_thor

readdir = os.fspath('P:/AHEC/DATA/THOR/')
savedir = os.fspath('P:/AHEC/Plots/THOR/')
#
chlist = []
chlist.extend(codestring('10??????00??AC^?', [['R']]))
chlist.extend(codestring('11SEBE????B?????'))
chlist.extend(codestring('11CHST0000THACX?'))
#chlist.extend(codestring('11CHST????THDC0?'))
chlist.extend(codestring('11CHST????THDSX?'))
chlist.extend(codestring('11SPIN??00THACX?'))
chlist.extend(codestring('11SPIN??00THAC??'))
chlist.extend(codestring('11THSP????THAV??'))
chlist.extend(codestring('11CLAV????TH???'))
chlist.extend(codestring('11NECKLO??THFO*?', [['X', 'Y']]))
#chlist.extend(codestring('11BRIC????TH????'))
chlist.extend(codestring('11PELV????THACX?'))
chlist.extend(codestring('11FEMRLE??THFOZ?'))
chlist.extend(codestring('11HEAD00??THAC*?', [['X','Y','Z']]))
#chlist.extend(codestring('11????????TH????'))
chlist.extend([ '11HEAD0000THAVXA', '11HEAD0000THAVYA', '11HEAD0000THAVZA', '11HEAD0000THANYA'])
### ---------------------------------------------------------------------------
#%%
THOR=False

#THOR subset:
#tcns = ['TC09-027', 'TC13-007', 'TC17-201', 'TC17-212', 'TC15-163', 'TC11-008', 'TC14-035', 'TC12-003', 'TC15-162', 'TC17-209', 'TC14-220', 'TC17-211', 'TC17-025', 'TC12-217', 'TC12-501', 'TC14-139', 'TC16-013', 'TC14-180', 'TC16-129', 'TC17-208']
#tcns = None
#Frontal Slip OK
#tcns = ['TC11-239', 'TC14-214', 'TC14-218', 'TC14-220', 'TC15-120', 'TC15-210', 'TC12-209', 'TC14-231', 'TC12-218', 'TC14-174', 'TC15-123', 'TC15-138', 'TC13-217', 'TC14-012', 'TC15-163', 'TC09-027', 'TC11-008', 'TC13-007', 'TC14-035', 'TC11-234', 'TC15-208', 'TC12-003', 'TC17-201', 'TC17-210', 'TC17-211', 'TC17-203', 'TC17-206', 'TC15-155', 'TC16-205', 'TC17-209', 'TC13-202', 'TC17-212', 'TC08-107', 'TC16-125', 'TC15-024', 'TC15-162', 'TC17-025', 'TC17-017', 'TC17-208', 'TC17-028', 'TC13-035', 'TC17-029', 'TC13-036', 'TC16-019', 'TC17-012', 'TC17-505', 'TC17-030', 'TC17-031']
#new slip/ok top 8
tcns = ['TC11-008', 'TC12-218', 'TC14-035', 'TC15-162', 'TC15-163', 'TC17-031', 'TC17-201', 'TC17-212', 'TC14-220', 'TC15-024', 'TC16-205', 'TC17-017', 'TC17-025', 'TC17-029', 'TC17-208', 'TC17-211']

writeHDF5(chlist, readdir)

if THOR:
    plot_thor(savedir, chlist, tcns)

print('All Done!')
