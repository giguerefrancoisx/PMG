# -*- coding: utf-8 -*-
"""
AHEC Basic Plotting Routine
    Used to quickly check channel traces for anomalies and outliers

Created on Fri Dec 1 12:15:00 2017

@author: giguerf
"""
def plotbook(subdir, tcns=None):
    """
    Docsting for basic plotting routine
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from PMG.COM import openbook as ob
    from PMG.COM import plotstyle as style

    readdir = os.fspath('P:/AHEC/SAI/')
    savedir = os.fspath('P:/AHEC/Plots/')

    descriptions = pd.read_excel('P:/AHEC/Descriptions.xlsx', index_col = 0)
    description = descriptions.to_dict()[descriptions.columns[0]]

    tcns = ['TC09-027', 'TC13-007', 'TC17-201', 'TC17-212', 'TC15-163', 'TC11-008', 'TC14-035', 'TC12-003', 'TC15-162', 'TC17-209', 'TC14-220', 'TC17-211', 'TC17-025', 'TC12-217', 'TC12-501', 'TC14-139', 'TC16-013', 'TC14-180', 'TC16-129', 'TC17-208']

#    REPLACE THE BELOW WITH HDF5 VERSION
#    time, slipdict, slidedict, okdict, singles, pairs = ob.thor(readdir, tcns)

    #%% FIGURE 1 - all pairs plotted individually

    xlim = (0, 0.3)
    xlabel = 'Time [s]'
#
    plt.close('all')
    for outer, inner in  [['11CLAVRIOUTHFOXA', '11CLAVRIINTHFOXA'],
                          ['11CLAVRIOUTHFOZA', '11CLAVRIINTHFOZA'],
                          ['11CLAVLEOUTHFOXA', '11CLAVLEINTHFOXA'],
                          ['11CLAVLEOUTHFOZA', '11CLAVLEINTHFOZA']]:

        slipinner = slipdict[inner]
        slipouter = slipdict[outer]
        okinner = okdict[inner]
        okouter = okdict[outer]
        plotinner = pd.concat([slipinner, okinner], axis=1).dropna(axis=1)
        plotouter = pd.concat([slipouter, okouter], axis=1).dropna(axis=1)
        plotinner = plotinner.loc[:,tcns] if tcns is not None else plotinner
        plotouter = plotouter.loc[:,tcns] if tcns is not None else plotouter
        if plotinner.shape[0] == 0:
            continue
        ylim = style.ylim_no_outliers(pd.concat([plotinner, plotouter], axis=1))
        ylabel = style.ylabel(inner[12:14], inner[14:15])

        r, c = style.sqfactors(len(plotinner.columns.tolist()))
        fig, axs = style.subplots(r, c, sharex='all', sharey='all', visible=True,
                                  num='All Pairs: '+inner, figsize=(5*c, 3.125*r))
        fig.suptitle('{ch} - {desc}'.format(ch = inner,
                     desc = description.get(inner, 'Description Unavailable')))

        for i, tcn in enumerate(plotinner.columns.tolist()):
            ax = axs[i]

            bool1 = tcn in slipinner.columns
            ax.plot(time, plotinner.loc[:, tcn], color = 'tab:blue' if bool1 else 'tab:orange', label = 'inner')
            ax.plot(time, plotouter.loc[:, tcn], color = 'tab:green' if bool1 else 'tab:purple', label = 'outer')
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            title = tcn
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.legend(loc=4)

        plt.tight_layout(rect=[0,0,1,0.92])
        plt.savefig(savedir+subdir+'All_Pairs_'+inner+'_combo.png', dpi=200)
        plt.close('all')

    #%% FIGURE 2 - Vehicle groups and means

    plt.close('all')
    for outer, inner in  [['11CLAVRIOUTHFOXA', '11CLAVRIINTHFOXA'],
                          ['11CLAVRIOUTHFOZA', '11CLAVRIINTHFOZA'],
                          ['11CLAVLEOUTHFOXA', '11CLAVLEINTHFOXA'],
                          ['11CLAVLEOUTHFOZA', '11CLAVLEINTHFOZA']]:

        slipinner = slipdict[inner]
        slipouter = slipdict[outer]
        okinner = okdict[inner]
        okouter = okdict[outer]
        plotinner = pd.concat([slipinner, okinner], axis=1).dropna(axis=1)
        plotouter = pd.concat([slipouter, okouter], axis=1).dropna(axis=1)
        plotinner = plotinner.loc[:,tcns] if tcns is not None else plotinner
        plotouter = plotouter.loc[:,tcns] if tcns is not None else plotouter

        ylim = style.ylim_no_outliers(pd.concat([plotinner, plotouter], axis=1))
        ylabel = style.ylabel(inner[12:14], inner[14:15])

        fig = plt.figure('Groups: '+inner, figsize=(20, 12.5))
        fig.suptitle('{ch} - {desc}'.format(ch = inner,
                     desc = description.get(inner, 'Description Unavailable')))

        #FIRST SUBPLOT - 'SLIP' Group full data set
        ax = plt.subplot(2,1,1)
        plt.plot(time, slipinner, '.', color = 'tab:blue', markersize=0.5, label = 'inner n = {}'.format(slipinner.shape[1]))
        plt.plot(time, slipouter, '.', color = 'tab:green', markersize=0.5, label = 'outer n = {}'.format(slipouter.shape[1]))
        plt.xlim(*xlim)
        ax.set_ylim(*ylim)
        plt.title('Slipping Belts')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        style.legend(ax=plt.gca(), loc=4)

        #THIRD SUBPLOT - 'OK' Group full data set
        plt.subplot(2,1,2, sharey = ax)
        plt.plot(time, okinner, '.', color = 'tab:orange', markersize=0.5, label = 'inner n = {}'.format(okinner.shape[1]))
        plt.plot(time, okouter, '.', color = 'tab:purple', markersize=0.5, label = 'outer n = {}'.format(okouter.shape[1]))
        plt.xlim(*xlim)
        ax.set_ylim(*ylim)
        plt.title('Ok Belts')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        style.legend(ax=plt.gca(), loc=4)

        plt.subplots_adjust(top=0.893, bottom=0.060, left=0.048, right=0.974, hspace=0.222, wspace=0.128)
        plt.savefig(savedir+subdir+'Belt_'+inner+'_combo.png', dpi=200)
        plt.close('all')
