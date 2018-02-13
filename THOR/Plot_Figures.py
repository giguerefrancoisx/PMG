# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:08:29 2018

@author: giguerf
"""

#:
subdir = 'test/'
tcns = None
chlist = ['10SIMELE00INACXD', '10SIMERI00INACXD', '10CVEHCG0000ACXD', '10CVEHCG0000ACYD', '10CVEHCG0000ACZD']
#chlist = [['10SIMELE00INACXD', '10SIMERI00INACXD'], ['10CVEHCG0000ACXD', '10CVEHCG0000ACYD', '10CVEHCG0000ACZD']]
chlist = [['10SIMELE00INACXD', '10SIMERI00INACXD', '10CVEHCG0000ACXD']]
project = 'AHEC'
#column = 'CBL_MODELE'
#column = 'PAIRS'
column = 'CIBLE'
#def plotbook(subdir, chlist, project, column, tcns=None):
if 1==1:
    """
    Docsting for basic plotting routine
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from GitHub.COM import openbook as ob
    from GitHub.COM import plotstyle as style

    readdir = os.fspath('P:/AHEC/SAI/')
    savedir = os.fspath('P:/AHEC/Plots/')

    descriptions = pd.read_excel('P:/AHEC/Descriptions.xlsx', index_col = 0)
    description = descriptions.to_dict()[descriptions.columns[0]]

    time, data = ob.openbook(readdir)
    if tcns is None:
        alldata = pd.concat(style.explode(data, []), axis=1)
        tcns = list(set(alldata.columns))
    table = ob.lookup_pairs(tcns, project)
#%%
    from cycler import cycler
    plt.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:green', 'tab:red'])

    plt.close('all')

    xlim = (0, 0.3)
    xlabel = 'Time [s]'

    if column == 'PAIRS':
        print('paired tests')
        table['PAIRS'] = table.CIBLE+' vs '+table.BELIER
        table['TCN'] = table['CIBLE']
        table2 = table.copy()
        table2['TCN'] = table2['BELIER']
        table2['CIBLE'] = table2['BELIER']
        table2['BELIER'] = table2['TCN']
        table = pd.concat([table,table2], axis=0)

    sets = set(table[column].sort_values())
    sub = {}
    for set_item in sets:
        sub[set_item] = table[table[column]==set_item]['CIBLE'].tolist()

    for channel in chlist:

        ch = channel[0] if isinstance(channel, list) else channel
        ylabel = style.ylabel(ch[12:14], ch[14:15])

        r, c = style.sqfactors(len(sub))
        fig, axs = style.subplots(r, c, sharex='all', sharey='all',
                                  num='All Pairs: '+ch, figsize=(5*c, 3.125*r))
        fig.suptitle('{ch} - {desc}'.format(ch = ch,
                     desc = description.get(ch, 'Description Unavailable')))

        chdata = pd.concat([data[ch] for ch in channel], axis=1) if isinstance(channel, list) else data[channel]
        ylim = style.ylim_no_outliers(chdata)

        for i, set_item in enumerate(sub):
            ax = axs[i]
            try:
                df = chdata.loc[:, sub[set_item]]
                for tcn in df:
                    ax.plot(time, df, '.', markersize=0.5)#, label=tcn) #TODO Fix labelling issue
            except KeyError:
                pass
            ax.plot(float('nan'), float('nan'), '.', label='n={}'.format(df.shape[1]), markersize=0)
#            ax.set_xlim(*xlim)
            ax.set_xlim(-0.01,0.3)
#            ax.set_ylim(*ylim)
            ax.set_ylim(-50,10)
            title = set_item
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            style.legend(ax, loc=4)

        plt.tight_layout(rect=[0,0,1,0.92])
        plt.savefig(savedir+subdir+'All_Pairs_'+ch+'.png', dpi=200)
        plt.close('all')
#%%
#    plt.close('all')
#    for channel in chlist:
#
#        ch = channel[0] if isinstance(channel, list) else channel
#        chdata = pd.concat([data[ch] for ch in channel], axis=1) if isinstance(channel, list) else data[channel]
#        ylim = style.ylim_no_outliers(chdata)
#
#        #
#        slip = slipdict[channel]
#        ok = okdict[channel]
#        if tcns is not None:
#            slip = slip.loc[:,tcns].dropna(axis=1)
#            ok = ok.loc[:,tcns].dropna(axis=1)
#
#        stats = {'Mean': slip.mean(axis = 1),
#                 'High': slip.mean(axis = 1)+2*slip.std(axis = 1),
#                 'Low': slip.mean(axis = 1)-2*slip.std(axis = 1)}
#        stats = pd.DataFrame(data = stats)
#        slipstats = stats
#
#        stats = {'Mean': ok.mean(axis = 1),
#                 'High': ok.mean(axis = 1)+2*ok.std(axis = 1),
#                 'Low': ok.mean(axis = 1)-2*ok.std(axis = 1)}
#        stats = pd.DataFrame(data = stats)
#        okstats = stats
#
#        ylim = style.ylim_no_outliers([slip, ok])
#        ylabel = style.ylabel(channel[12:14], channel[14:15])
#
#        fig = plt.figure('Groups: '+channel, figsize=(20, 12.5))
#        fig.suptitle('{ch} - {desc}'.format(ch = channel,
#                     desc = description.get(channel, 'Description Unavailable')))
#
#        #FIRST SUBPLOT - 'SLIP' Group full data set
#        ax = plt.subplot(2,2,1)
#        plt.plot(time, slip, '.', color = 'tab:blue', markersize=0.5, label = 'n = {}'.format(slip.shape[1]))
#        plt.xlim(*xlim)
#        ax.set_ylim(*ylim)
#        plt.title('Slipping Belts')
#        plt.ylabel(ylabel)
#        plt.xlabel(xlabel)
#        style.legend(ax=plt.gca(), loc=4)
#
#        #THIRD SUBPLOT - 'OK' Group full data set
#        plt.subplot(2,2,3, sharey = ax)
#        plt.plot(time, ok, '.', color = 'tab:orange', markersize=0.5, label = 'n = {}'.format(ok.shape[1]))
#        plt.xlim(*xlim)
#        ax.set_ylim(*ylim)
#        plt.title('Ok Belts')
#        plt.ylabel(ylabel)
#        plt.xlabel(xlabel)
#        style.legend(ax=plt.gca(), loc=4)
#
#        #FOURTH SUBPLOT - 'CIBLE' vs 'BELIER' Groups (mean and intervals)
#        plt.subplot(2,2,2)
#        plt.plot(time, slipstats['Mean'], color = 'tab:blue', label = 'Mean (Slip), n = {}'.format(slip.shape[1]))
#        plt.fill_between(time, slipstats['High'], slipstats['Low'], color = 'tab:blue', alpha = 0.25, label = 'Intervals (Slip)')
#        plt.plot(time, okstats['Mean'], color = 'tab:orange', label = 'Mean (Ok), n = {}'.format(ok.shape[1]))
#        plt.fill_between(time, okstats['High'], okstats['Low'], color = 'tab:orange', alpha = 0.25, label = 'Intervals (Ok)')
#        plt.xlim(*xlim)
#        ax.set_ylim(*ylim)
#        plt.title('All Belts (Mean and Intervals)')
#        plt.ylabel(ylabel)
#        plt.xlabel(xlabel)
#        plt.legend(loc = 4)
#
#        plt.subplots_adjust(top=0.893, bottom=0.060, left=0.048, right=0.974, hspace=0.222, wspace=0.128)
#        plt.savefig(savedir+subdir+'Belt_'+channel+'.png', dpi=200)
#        plt.close('all')



















#%%
#plotbook(subdir, chlist, project, column, tcns=None)