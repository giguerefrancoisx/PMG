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
    from PMG.COM import openbook as ob
    from PMG.COM import plotstyle as style
    from PMG.COM import data

    descriptions = pd.read_excel('P:/AHEC/Descriptions.xlsx', index_col = 0)
    description = descriptions.to_dict()[descriptions.columns[0]]

    time, categories = ob.thor(chlist, tcns)
    slip, ok = categories['SLIP'], categories['OK']

    #%% FIGURE 1 - all pairs plotted individually

    xlim = (0, 0.3)
    xlabel = 'Time [s]'
#
    plt.close('all')
    for channel in chlist:
        slipdf = slip[channel]
        okdf = ok[channel]
        plotdata = pd.concat([slipdf, okdf], axis=1)

        if plotdata.empty:
            continue
        ylim = style.ylim_no_outliers(plotdata)
        ylabel = style.ylabel(channel[12:14], channel[14:15])

        r, c = style.sqfactors(len(plotdata.columns.tolist()))
        fig, axs = style.subplots(r, c, sharex='all', sharey='all',
                                  visible=True, num='All Pairs: '+channel)
        fig.suptitle('{ch} - {desc}'.format(ch = channel,
                     desc = description.get(channel, 'Description Unavailable')))

        for i, tcn in enumerate(plotdata.columns.tolist()):
            ax = axs[i]
            ax.plot(time, plotdata.loc[:, tcn], color = 'tab:green', label = tcn)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            title = tcn
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

        slipdf, slipstats = slip[channel], data.stats(slip[channel])
        okdf, okstats = ok[channel], data.stats(ok[channel])

        ylim = style.ylim_no_outliers([slipdf, okdf])
        ylabel = style.ylabel(channel[12:14], channel[14:15])

        fig = plt.figure('Groups: '+channel, figsize=(20, 12.5))
        fig.suptitle('{ch} - {desc}'.format(ch = channel,
                     desc = description.get(channel, 'Description Unavailable')))

        #FIRST SUBPLOT - 'SLIP' Group full data set
        ax = plt.subplot(2,2,1)
        plt.plot(time, slipdf, '.', color = 'tab:blue', markersize=0.5, label = 'n = {}'.format(slipdf.shape[1]))
        plt.xlim(*xlim)
        ax.set_ylim(*ylim)
        plt.title('Slipping Belts')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        style.legend(ax=plt.gca(), loc=4)

        #THIRD SUBPLOT - 'OK' Group full data set
        plt.subplot(2,2,3, sharey = ax)
        plt.plot(time, okdf, '.', color = 'tab:orange', markersize=0.5, label = 'n = {}'.format(okdf.shape[1]))
        plt.xlim(*xlim)
        ax.set_ylim(*ylim)
        plt.title('Ok Belts')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        style.legend(ax=plt.gca(), loc=4)

        #FOURTH SUBPLOT - 'CIBLE' vs 'BELIER' Groups (mean and intervals)
        plt.subplot(2,2,2, sharey = ax)
        plt.plot(time, slipstats['Mean'], color = 'tab:blue', label = 'Mean (Slip), n = {}'.format(slipdf.shape[1]))
        plt.plot(time, slipstats['Mean-between'], color = 'tab:purple', label = 'Mean-between (Slip), n = {}'.format(slipdf.shape[1]))
        plt.fill_between(time, slipstats['High'], slipstats['Low'], color = 'tab:blue', alpha = 0.25, label = 'Intervals (Slip)')
        plt.plot(time, okstats['Mean'], color = 'tab:orange', label = 'Mean (Ok), n = {}'.format(okdf.shape[1]))
        plt.plot(time, okstats['Mean-between'], color = 'tab:red', label = 'Mean-between (Ok), n = {}'.format(okdf.shape[1]))
        plt.fill_between(time, okstats['High'], okstats['Low'], color = 'tab:orange', alpha = 0.25, label = 'Intervals (Ok)')
        plt.xlim(*xlim)
        ax.set_ylim(*ylim)
        plt.title('All Belts (Mean and Intervals)')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend(loc = 4)

        plt.subplots_adjust(top=0.893, bottom=0.060, left=0.048, right=0.974, hspace=0.222, wspace=0.128)
        plt.savefig(savedir+'Belt_'+channel+'.png', dpi=200)
        plt.close('all')
