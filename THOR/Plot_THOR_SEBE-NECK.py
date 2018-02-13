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
#    import pandas as pd
    import matplotlib.pyplot as plt
    from PMG.COM import openbook as ob
    from PMG.COM import plotstyle as style

    readdir = os.fspath('P:/AHEC/SAI/')
    savedir = os.fspath('P:/AHEC/Plots/')

    time, slipdict, slidedict, okdict, singles, pairs = ob.thor(readdir, tcns)
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
    lines = {'TC11-008': [0.016, 0.062, 0.080, 0.093],
             'TC12-218': [0.030, 0.063, 0.082, 0.083],
             'TC14-035': [0.042, 0.070, 0.081, 0.088],
             'TC15-162': [0.013, 0.077, 0.088, 0.095],
             'TC15-163': [0.000, 0.000, 0.000, 0.092],
             'TC17-031': [0.030, 0.060, 0.076, 0.083],
             'TC17-201': [0.030, 0.056, 0.076, 0.084],
             'TC17-212': [0.031, 0.060, 0.080, 0.090]}

    #%% FIGURE 1 - all pairs plotted individually

    xlim = (0, 0.15)
    xlabel = 'Time [s]'
#
    plt.close('all')

    r, c = style.sqfactors(len(tcns))
    fig, axs = style.subplots(r, c, sharex='all', sharey='all', visible=True,
                              num='SEBE-NECK', figsize=(5*c, 3.125*r))

    for i, tcn in enumerate(tcns):
        ax = axs[i]

        slip_cols = []
        for channel in slipdict:
            slip_cols.extend(slipdict[channel].columns)
        slip_cols = list(set(slip_cols))
        slip = tcn in slip_cols
        ax.plot(time, slipdict['11SEBE0000B3FO0D'][tcn] if slip else okdict['11SEBE0000B3FO0D'][tcn], color='tab:blue')
        ax2 = plt.twinx(ax)
        ax2.plot(time, slipdict['11NECKLO00THFOXA'][tcn] if slip else okdict['11NECKLO00THFOXA'][tcn], color='tab:green')
        ax2.plot(time, slipdict['11NECKLO00THFOYA'][tcn] if slip else okdict['11NECKLO00THFOYA'][tcn], color='tab:green')
        ax3 = plt.twinx(ax)
        ax3.plot(time, slipdict['11CLAVLEINTHFOZA'][tcn] if slip else okdict['11CLAVLEINTHFOZA'][tcn], color='tab:orange')
        ax3.plot(time, slipdict['11CLAVRIOUTHFOZA'][tcn] if slip else okdict['11CLAVRIOUTHFOZA'][tcn], color='tab:purple')
        for line, color in zip(lines.get(tcn, [-1]*4), colors):
            ax.axvline(line, color=color)
        ax.set_xlim(*xlim)
        ax.set_ylim(-6000,6000)
        ax2.set_ylim(-2000,2000)
        ax3.set_ylim(-1000,1000)

        title = tcn
        ax.set_title(title)
        ax.set_ylabel('Seat Belt Tension [N]')
        ax2.set_ylabel('Neck Force [N]')
        ax.set_xlabel(xlabel)
        ax.legend(loc=4)

    plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig(savedir+subdir+'SEBE-NECK.png', dpi=200)
    plt.close('all')

    #%% FIGURE 2 - Vehicle groups and means

    plt.close('all')
    plt.figure('SEBE-NECK Groups', figsize=(20, 12.5))

    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(2,3,1)
    plt.plot(time, slipdict['11SEBE0000B3FO0D'], '.', color = 'tab:blue', markersize=0.5, label = 'inner n = {}'.format(len(slipdict['11SEBE0000B3FO0D'])))
    plt.xlim(*xlim)
    ax.set_ylim(-6000,6000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('Slip SEBE')
    plt.ylabel('Seat Belt Tension [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)
    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(2,3,2)
    plt.plot(time, slipdict['11NECKLO00THFOXA'], '.', color = 'tab:orange', markersize=0.5, label = 'inner n = {}'.format(len(slipdict['11NECKLO00THFOXA'])))
    plt.xlim(*xlim)
    ax.set_ylim(-2000,2000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('Slip NECK X')
    plt.ylabel('Neck Force [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)
    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(2,3,3)
    plt.plot(time, slipdict['11NECKLO00THFOYA'], '.', color = 'tab:purple', markersize=0.5, label = 'inner n = {}'.format(len(slipdict['11NECKLO00THFOYA'])))
    plt.xlim(*xlim)
    ax.set_ylim(-2000,2000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('Slip NECK Y')
    plt.ylabel('Neck Force [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)
    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(2,3,4)
    plt.plot(time, okdict['11SEBE0000B3FO0D'], '.', color = 'tab:blue', markersize=0.5, label = 'inner n = {}'.format(len(okdict['11SEBE0000B3FO0D'])))
    plt.xlim(*xlim)
    ax.set_ylim(-6000,6000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('OK SEBE')
    plt.ylabel('Seat Belt Tension [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)
    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(2,3,5)
    plt.plot(time, okdict['11NECKLO00THFOXA'], '.', color = 'tab:orange', markersize=0.5, label = 'inner n = {}'.format(len(okdict['11NECKLO00THFOXA'])))
    plt.xlim(*xlim)
    ax.set_ylim(-2000,2000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('OK NECK X')
    plt.ylabel('Neck Force [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)
    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(2,3,6)
    plt.plot(time, okdict['11NECKLO00THFOYA'], '.', color = 'tab:purple', markersize=0.5, label = 'inner n = {}'.format(len(okdict['11NECKLO00THFOYA'])))
    plt.xlim(*xlim)
    ax.set_ylim(-2000,2000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('OK NECK Y')
    plt.ylabel('Neck Force [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)

    plt.subplots_adjust(top=0.893, bottom=0.060, left=0.048, right=0.974, hspace=0.222, wspace=0.128)
    plt.savefig(savedir+subdir+'SEBE-NECK_stats.png', dpi=200)
    plt.close('all')
    #%% FIGURE 2 - Vehicle groups and means

    plt.close('all')
    plt.figure('SEBE-NECK Groups', figsize=(20, 12.5))

    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(3,2,1)
    plt.plot(time, slipdict['11SEBE0000B3FO0D'], '.', color = 'tab:blue', markersize=0.5, label = 'inner n = {}'.format(len(slipdict['11SEBE0000B3FO0D'])))
    plt.xlim(*xlim)
    ax.set_ylim(-6000,6000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('Slip SEBE')
    plt.ylabel('Seat Belt Tension [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)
    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(3,2,3)
    plt.plot(time, slipdict['11NECKLO00THFOXA'], '.', color = 'tab:orange', markersize=0.5, label = 'inner n = {}'.format(len(slipdict['11NECKLO00THFOXA'])))
    plt.xlim(*xlim)
    ax.set_ylim(-2000,2000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('Slip NECK X')
    plt.ylabel('Neck Force [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)
    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(3,2,5)
    plt.plot(time, slipdict['11NECKLO00THFOYA'], '.', color = 'tab:purple', markersize=0.5, label = 'inner n = {}'.format(len(slipdict['11NECKLO00THFOYA'])))
    plt.xlim(*xlim)
    ax.set_ylim(-2000,2000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('Slip NECK Y')
    plt.ylabel('Neck Force [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)
    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(3,2,2)
    plt.plot(time, okdict['11SEBE0000B3FO0D'], '.', color = 'tab:blue', markersize=0.5, label = 'inner n = {}'.format(len(okdict['11SEBE0000B3FO0D'])))
    plt.xlim(*xlim)
    ax.set_ylim(-6000,6000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('OK SEBE')
    plt.ylabel('Seat Belt Tension [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)
    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(3,2,4)
    plt.plot(time, okdict['11NECKLO00THFOXA'], '.', color = 'tab:orange', markersize=0.5, label = 'inner n = {}'.format(len(okdict['11NECKLO00THFOXA'])))
    plt.xlim(*xlim)
    ax.set_ylim(-2000,2000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('OK NECK X')
    plt.ylabel('Neck Force [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)
    #FIRST SUBPLOT - 'SLIP' Group full data set
    ax = plt.subplot(3,2,6)
    plt.plot(time, okdict['11NECKLO00THFOYA'], '.', color = 'tab:purple', markersize=0.5, label = 'inner n = {}'.format(len(okdict['11NECKLO00THFOYA'])))
    plt.xlim(*xlim)
    ax.set_ylim(-2000,2000)
    ax.axvline(0.055)
    ax.axvline(0.080)
    ax.axvline(0.090)
    plt.title('OK NECK Y')
    plt.ylabel('Neck Force [N]')
    plt.xlabel(xlabel)
    style.legend(ax=plt.gca(), loc=4)

    plt.subplots_adjust(top=0.893, bottom=0.060, left=0.048, right=0.974, hspace=0.222, wspace=0.128)
    plt.savefig(savedir+subdir+'SEBE-NECK_stats2.png', dpi=200)
    plt.close('all')
