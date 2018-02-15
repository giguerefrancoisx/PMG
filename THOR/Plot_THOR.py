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
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from PMG.COM import openbook as ob
    from PMG.COM import plotstyle as style

    readdir = os.fspath('P:/AHEC/SAI/')
#    savedir = os.fspath('P:/AHEC/Plots/')

    descriptions = pd.read_excel('P:/AHEC/Descriptions.xlsx', index_col = 0)
    description = descriptions.to_dict()[descriptions.columns[0]]

    time, slipdict, okdict, singles = ob.thorHDF5(chlist, tcns)

    #%% FIGURE 1 - all pairs plotted individually

    xlim = (0, 0.3)
    xlabel = 'Time [s]'
#
    plt.close('all')
    for channel in chlist:
        slip = slipdict[channel]
        ok = okdict[channel]
        plotdata = pd.concat([slip, ok], axis=1)
        plotdata = plotdata.loc[:,tcns] if tcns is not None else plotdata
        if plotdata.shape[0] == 0:
            continue
        ylim = style.ylim_no_outliers(plotdata)
        ylabel = style.ylabel(channel[12:14], channel[14:15])

        r, c = style.sqfactors(len(plotdata.columns.tolist()))
        fig, axs = style.subplots(r, c, sharex='all', sharey='all', visible=True,
                                  num='All Pairs: '+channel, figsize=(5*c, 3.125*r))
        fig.suptitle('{ch} - {desc}'.format(ch = channel,
                     desc = description.get(channel, 'Description Unavailable')))

#        data = pd.concat([cib,bel], axis=1)
        for i, tcn in enumerate(plotdata.columns.tolist()):
            ax = axs[i]

#            ax.plot(time, cib.loc[:, cib_tcn], color = 'tab:blue', label = cib_tcn)
#            ax.plot(time, bel.loc[:, bel_tcn], color = 'tab:green', label = bel_tcn)
            ax.plot(time, plotdata.loc[:, tcn], color = 'tab:green', label = tcn)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

#            title = '{} vs {}: {} vs {}'.format(cib_tcn, pairs.loc[cib_tcn,'BELIER'],
#                     pairs.loc[cib_tcn,'CBL_MODELE'], pairs.loc[cib_tcn,'BLR_MODELE'])
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
        slip = slipdict[channel]
        ok = okdict[channel]
        if tcns is not None:
            slip = slip.loc[:,tcns].dropna(axis=1)
            ok = ok.loc[:,tcns].dropna(axis=1)

        stats = {'Mean': slip.mean(axis = 1),
                 'High': slip.mean(axis = 1)+2*slip.std(axis = 1),
                 'Low': slip.mean(axis = 1)-2*slip.std(axis = 1)}
        stats = pd.DataFrame(data = stats)
        slipstats = stats

        stats = {'Mean': ok.mean(axis = 1),
                 'High': ok.mean(axis = 1)+2*ok.std(axis = 1),
                 'Low': ok.mean(axis = 1)-2*ok.std(axis = 1)}
        stats = pd.DataFrame(data = stats)
        okstats = stats

        ylim = style.ylim_no_outliers([slip, ok])
        ylabel = style.ylabel(channel[12:14], channel[14:15])

        fig = plt.figure('Groups: '+channel, figsize=(20, 12.5))
        fig.suptitle('{ch} - {desc}'.format(ch = channel,
                     desc = description.get(channel, 'Description Unavailable')))

        #FIRST SUBPLOT - 'SLIP' Group full data set
        ax = plt.subplot(2,2,1)
        plt.plot(time, slip, '.', color = 'tab:blue', markersize=0.5, label = 'n = {}'.format(slip.shape[1]))
        plt.xlim(*xlim)
        ax.set_ylim(*ylim)
        plt.title('Slipping Belts')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        style.legend(ax=plt.gca(), loc=4)

        #THIRD SUBPLOT - 'OK' Group full data set
        plt.subplot(2,2,3, sharey = ax)
        plt.plot(time, ok, '.', color = 'tab:orange', markersize=0.5, label = 'n = {}'.format(ok.shape[1]))
        plt.xlim(*xlim)
        ax.set_ylim(*ylim)
        plt.title('Ok Belts')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        style.legend(ax=plt.gca(), loc=4)

        #FOURTH SUBPLOT - 'CIBLE' vs 'BELIER' Groups (mean and intervals)
        plt.subplot(2,2,2)
        plt.plot(time, slipstats['Mean'], color = 'tab:blue', label = 'Mean (Slip), n = {}'.format(slip.shape[1]))
        plt.fill_between(time, slipstats['High'], slipstats['Low'], color = 'tab:blue', alpha = 0.25, label = 'Intervals (Slip)')
        plt.plot(time, okstats['Mean'], color = 'tab:orange', label = 'Mean (Ok), n = {}'.format(ok.shape[1]))
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
