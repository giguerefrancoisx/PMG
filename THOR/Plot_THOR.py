# -*- coding: utf-8 -*-
"""
AHEC Basic Plotting Routine
    Used to quickly check channel traces for anomalies and outliers

Created on Fri Dec 1 12:15:00 2017

@author: giguerf
"""
def plotbook(savedir, chlist, tcns=None):
    """
    Plot THOR-related basic overview plots. All pairs by channels and both
    groups with means/intervals by channels.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from PMG.COM import plotstyle as style, data, table as tb

    descriptions = pd.read_excel('P:/AHEC/Descriptions.xlsx', index_col = 0)
    description = descriptions.to_dict()[descriptions.columns[0]]

    time, fulldata = data.import_data('P:/AHEC/DATA/THOR/', chlist, tcns, check=False)
#    for k,v in fulldata.items():
#        fulldata[k] = data.check_and_clean(v, stage=2)

    table = tb.get('THOR')
    table = table[table.CIBLE.isin(tcns)]
    slips  = table[table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
    oks = table[table.CBL_BELT.isin(['OK'])].CIBLE.tolist()
    table = table[table.CIBLE.isin(slips+oks)]

    #%% FIGURE 1 - all pairs plotted individually

    xlim = (0, 0.3)
    xlabel = 'Time [s]'

    plt.close('all')
    for channel in chlist:
        slipdf = fulldata[channel].loc[:,slips]
        okdf = fulldata[channel].loc[:,oks]
        plotdata = pd.concat([slipdf, okdf], axis=1)

        if plotdata.empty:
            continue

        ylabel = style.ylabel(channel[12:14], channel[14:15])

        r, c = style.sqfactors(len(plotdata.columns.tolist()))
        fig, axs = style.subplots(r, c, sharex='all', sharey='all',
                                  visible=True, num='All Pairs: '+channel)
        fig.suptitle('{ch} - {desc}'.format(ch = channel,
                     desc = description.get(channel, 'Description Unavailable')))

        for i, tcn in enumerate(table.CIBLE):
            ax = axs[i]
            ax.plot(time, plotdata.loc[:, tcn], color = 'tab:green', label = tcn)
            ax.set_xlim(*xlim)

            title = ' '.join([tcn,table[table.CIBLE==tcn].CBL_MODELE.tolist()[0]])
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.legend(loc=4)

        plt.tight_layout(rect=[0,0,1,0.92])
        plt.savefig(savedir+'All_Pairs_'+channel+'.png', dpi=200)
        plt.close('all')

    #%% FIGURE 2 - Vehicle groups and means

    plt.close('all')
    for channel in chlist:

        slipdf = fulldata[channel].loc[:,slips]
        okdf = fulldata[channel].loc[:,oks]

        ylabel = style.ylabel(channel[12:14], channel[14:15])

        fig, axs = style.subplots(2,2,sharex='all',sharey='all', num='Belt: '+channel, figsize=(20, 12.5))
        fig.suptitle('{ch} - {desc}'.format(ch = channel,
                     desc = description.get(channel, 'Description Unavailable')))

        #FIRST SUBPLOT - 'SLIP' Group full data set
        ax = axs[0]
        ax.plot(time, slipdf, lw=1, color = 'tab:blue', label = 'n = {}'.format(slipdf.shape[1]))
        ax.set_title('Slipping Belts')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        style.legend(ax, loc=4)

        #THIRD SUBPLOT - 'OK' Group full data set
        ax = axs[2]
        ax.plot(time, okdf, lw=1, color = 'tab:orange', label = 'n = {}'.format(okdf.shape[1]))
        ax.set_title('Ok Belts')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        style.legend(ax, loc=4)

        #SECOND SUBPLOT - 'CIBLE' vs 'BELIER' Groups (median and intervals)
        ax = axs[1]
        data.tolerance(ax, time, slipdf, 'tab:blue')
        data.tolerance(ax, time, okdf, 'tab:orange')

        ax.set_title('All Belts (Mean and Intervals)')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend(loc=4)

        ax.set_xlim(*xlim)
#%%
        plt.subplots_adjust(top=0.893, bottom=0.060, left=0.048, right=0.974, hspace=0.222, wspace=0.128)
        plt.savefig(savedir+'Belt_'+channel+'.png', dpi=200)
        plt.close('all')
