# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:53:00 2018

@author: giguerf
"""
import matplotlib.pyplot as plt
from PMG.COM.writebook import writeHDF5
from PMG.COM.openbook import openHDF5
from PMG.COM import table as tb, plotsyle as style

table = tb.get('SLED')

SLED = 'P:/SLED/Data/'
chlist = ['S0SLED000000ACXD',
          '12HEAD0000Y7ACXA','12HEAD0000Y2ACXA',
          '12CHST0000Y7ACXC','12CHST0000Y2ACXC',
          '12PELV0000Y7ACXA','12PELV0000Y2ACXA']

writeHDF5(SLED, chlist)
#%%
time, fulldata = openHDF5(SLED, chlist)

"""NO PELVIS Y2:
SE16-0250, SE16-0251, SE16-0253, SE16-0366, SE16-0367, SE16-0374, SE16-0375, SE16-0376, SE16-0377, SE16-0394, SE16-0398, SE16-0409
"""
#%%
chname = dict(zip(chlist, ['SLED','HEAD','HEAD','CHEST','CHEST','PELVIS','PELVIS']))
tcns = table[table.MODEL=='BABY TREND blue  HIII-6-YR'].SEAT.tolist()

plt.close('all')
for tcns in [tcns]:
    fig, axs = style.subplots(2,2,sharex='all')
    for channels, ax in zip([['12HEAD0000Y7ACXA','12HEAD0000Y2ACXA'],['12CHST0000Y7ACXC','12CHST0000Y2ACXC'],['12PELV0000Y7ACXA','12PELV0000Y2ACXA']], axs):
        for ch in channels:
            try:
                for tcn in tcns:
                    sled = table[table.SEAT==tcn].SLED.tolist()[0]
                    ax.plot(time, fulldata[ch].loc[:,tcn], label=sled)
            except KeyError:
                pass
        ax.set_title(chname[ch])
    axs[-1].set_xlim(0,0.2)
    axs[0].legend()
    plt.tight_layout()