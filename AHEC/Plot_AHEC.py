# -*- coding: utf-8 -*-
"""
AHEC Basic Plotting Routine
    Used to quickly check channel traces for anomalies and outliers

Created on Fri Dec 1 12:15:00 2017

@author: giguerf
"""
def plotbook(subdir, chlist, tcns=None):
#if 1==1:
    """
    Docsting for basic plotting routine
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from GitHub.COM import openbook as ob
    from GitHub.COM import plotstyle as style

    readdir = os.fspath('P:/AHEC/SAI/')
    savedir = os.fspath('P:/AHEC/Plots/')

    descriptions = pd.read_excel('P:/AHEC/Descriptions.xlsx', index_col = 0)
    description = descriptions.to_dict()[descriptions.columns[0]]

    time, cibdict, beldict, pairs = ob.cibbel(readdir)

    xlim = (0, 0.3)
    xlabel = 'Time [s]'
    colors = {'cib': '', 'bel': ''}
    #%% FIGURE 1 - all pairs plotted individually

    plt.close('all')

    for channel in chlist:
        if tcns is None:
            cib = cibdict[channel]
            bel = beldict[channel]
            pairs = ob.lookup_pairs(cib.columns, project='AHEC')
#            pairs = ob.lookup_pairs(cib.columns.tolist(), project='AHEC')
#            pairs = ob.lookup_pairs(project='AHEC') #move out of if with tcns arg
        else:
            cib = cibdict[channel].loc[:,tcns].dropna(axis=1)
            bel = beldict[channel].loc[:,tcns].dropna(axis=1)
            pairs = ob.lookup_pairs(tcns, project='AHEC')



        if cib.shape[0] == 0:
            continue

        plotframes = [cib, bel]
        ylim = style.ylim_no_outliers(plotframes)
        if (channel[2:6] == 'SIME') or (channel[2:6] == 'CVEH' and channel[14:15] == 'X'):
            ylim = (-60, 30)
        ylabel = style.ylabel(channel[12:14], channel[14:15])

#        pairs = ob.lookup_pairs(cib.columns.tolist(), project='AHEC')
#        pairs = ob.lookup_pairs(tcns, project='AHEC')
#        pairs = pairs.set_index('CIBLE')
        r, c = style.sqfactors(pairs.shape[0])
        fig, axs = style.subplots(r, c, sharex='all', sharey='all', visible=True,
                                  num='All Pairs: '+channel, figsize=(5*c, 3.125*r))
        fig.suptitle(
                '{sbdir} - Frontal Offset Crash Test at {speed} km/h\n{ch} - {desc}'.format(
                sbdir = subdir[:-4], speed = subdir[-3:-1], ch = channel,
                desc = description.get(channel, 'Description Unavailable')))

#        for i, (cib_tcn, bel_tcn) in enumerate(zip(cib.columns, bel.columns)):
        for i, (cib_tcn, bel_tcn) in enumerate(zip(pairs.CIBLE, pairs.BELIER)):
            ax = axs[i]

            pair=pairs[pairs.CIBLE==cib_tcn]
#            bool1 = pairs.loc[cib_tcn,'SUBSET'] == 'HEV vs ICE'
            bool1 = pair.SUBSET.item() == 'HEV vs ICE'
#            bool2 = not pairs.loc[cib_tcn,'SUBSET'] == 'General'
            bool2 = not pair.SUBSET.item() == 'General'
            colors['cib'] = ('tab:blue' if bool1 else 'tab:purple') if bool2 else 'tab:pink'
            colors['bel'] = ('tab:orange' if bool1 else 'tab:green') if bool2 else 'tab:grey'

#            ax.plot(time, cib.loc[:, cib_tcn], color=colors['cib'], label=cib_tcn)
#            ax.plot(time, bel.loc[:, bel_tcn], color=colors['bel'], label=bel_tcn)
            ax.plot(time, cib.get(cib_tcn, default=time*np.nan), color=colors['cib'], label=cib_tcn)
            ax.plot(time, bel.get(bel_tcn, default=time*np.nan), color=colors['bel'], label=bel_tcn)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            title = '{} vs {}: {} vs {}'.format(cib_tcn, pair.BELIER.to_string(index=False),
                     pair.CBL_MODELE.to_string(index=False), pair.BLR_MODELE.to_string(index=False))
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.legend(loc=4)

        plt.tight_layout(rect=[0,0,1,0.92])
        plt.savefig(savedir+subdir+'All_Pairs_'+channel+'.png', dpi=200)
        plt.close('all')

    #%% FIGURE 3 - Vehicle groups

    pairs = ob.lookup_pairs(tcns, project = 'AHEC')
    popdict = ob.populations(readdir, pairs)

    NULLDF = pd.DataFrame(columns=tcns)
    NULL = {'CIBLE':NULLDF,'BELIER':NULLDF}

    plt.close('all')
    for channel in chlist:
        HEV = popdict[channel].get('HEV vs ICE', NULL)['CIBLE']
        ICE = popdict[channel].get('HEV vs ICE', NULL)['BELIER']
        OLD = popdict[channel].get('Old vs New', NULL)['CIBLE']
        NEW = popdict[channel].get('Old vs New', NULL)['BELIER']
        CIB = popdict[channel].get('General', NULL)['CIBLE']
        BEL = popdict[channel].get('General', NULL)['BELIER']

        if tcns is not None:
            HEV = pd.DataFrame(HEV, columns=tcns).dropna(axis=1)
            ICE = pd.DataFrame(ICE, columns=tcns).dropna(axis=1)
            OLD = pd.DataFrame(OLD, columns=tcns).dropna(axis=1)
            NEW = pd.DataFrame(NEW, columns=tcns).dropna(axis=1)
            CIB = pd.DataFrame(CIB, columns=tcns).dropna(axis=1)
            BEL = pd.DataFrame(BEL, columns=tcns).dropna(axis=1)

#        HEVstats = popdict['HEV vs ICE']['CIBLE'][channel+'_stats']
#        ICEstats = popdict['HEV vs ICE']['BELIER'][channel+'_stats']
#        OLDstats = popdict['Old vs New']['CIBLE'][channel+'_stats']
#        NEWstats = popdict['Old vs New']['BELIER'][channel+'_stats']
#        CIBstats = popdict['General']['CIBLE'][channel+'_stats']
#        BELstats = popdict['General']['BELIER'][channel+'_stats']

        fig = plt.figure('Groups: '+channel, figsize=(20, 12.5))
        fig.suptitle(
                '{sbdir} - Frontal Offset Crash Test at {speed} km/h\n{ch} - {desc}'.format(
                sbdir = subdir[:-4], speed = subdir[-3:-1], ch = channel,
                desc = description.get(channel, 'Description Unavailable')))

        plotframes = [CIB, BEL, HEV, ICE, OLD, NEW]
        ylim = style.ylim_no_outliers(plotframes)
        ylabel = style.ylabel(channel[12:14], channel[14:15])

        if not CIB.empty:
            #FIRST SUBPLOT - 'CIBLE' Group full data set
            ax = plt.subplot(2,3,1)
            plt.plot(time, CIB, '.', color = 'tab:pink', markersize=0.5,
                     label = 'n = {}'.format(CIB.shape[1]))
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.title('Cible Vehicle Data')
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            style.legend(ax=plt.gca(), loc=4)

        if not BEL.empty:
            #SECOND SUBPLOT - 'BELIER' Group full data set
            plt.subplot(2,3,4, sharey = ax)
            plt.plot(time, BEL, '.', color = 'tab:gray', markersize=0.5,
                     label = 'n = {}'.format(BEL.shape[1]))
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.title('Belier Vehicle Data')
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            style.legend(ax=plt.gca(), loc=4)

        if not HEV.empty:
            # HEV
            plt.subplot(2,3,2, sharey = ax)
            plt.plot(time, HEV, '.', color = 'tab:blue', markersize=0.5,
                     label = 'n = {}'.format(HEV.shape[1]))
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.title('HEV Vehicle Data')
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            style.legend(ax=plt.gca(), loc=4)

        if not ICE.empty:
            # ICE
            plt.subplot(2,3,5, sharey = ax)
            plt.plot(time, ICE, '.', color = 'tab:orange', markersize=0.5,
                     label = 'n = {}'.format(ICE.shape[1]))
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.title('ICE Vehicle Data')
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            style.legend(ax=plt.gca(), loc=4)

        if not NEW.empty:
            # NEW
            plt.subplot(2,3,3, sharey = ax)
            plt.plot(time, NEW, '.', color = 'tab:green', markersize=0.5,
                     label = 'n = {}'.format(NEW.shape[1]))
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.title('New Vehicle Data')
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            style.legend(ax=plt.gca(), loc=4)

        if not OLD.empty:
            # OLD
            plt.subplot(2,3,6, sharey = ax)
            plt.plot(time, OLD, '.', color = 'tab:purple', markersize=0.5,
                     label = 'n = {}'.format(OLD.shape[1]))
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.title('Old Vehicle Data')
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            style.legend(ax=plt.gca(), loc=4)

        plt.tight_layout(rect=[0,0,1,0.92])
        plt.savefig(savedir+subdir+'Groups_'+channel+'.png', dpi=200)
        plt.close('all')

    #%% FIGURE 4 - all pairs plotted individually

    plt.close('all')

    table = pd.read_excel('P:/AHEC/ahectable.xlsx')
    table = table.dropna(axis=0, how='all').dropna(axis=1, thresh=2)

    for channel in chlist:
        if tcns is None:
            cib = cibdict[channel]
            bel = beldict[channel]
            table2 = table[table.loc[:,['CIBLE','BELIER']].isin(cib.columns).any(axis=1)]
        else:
            cib = cibdict[channel].loc[:,tcns].dropna(axis=1)
            bel = beldict[channel].loc[:,tcns].dropna(axis=1)
            table2 = table[table.loc[:,['CIBLE','BELIER']].isin(tcns).any(axis=1)]

        pairs = table2.loc[:,['CIBLE','BELIER','VITESSE','CBL_MODELE','BLR_MODELE',
                         'SUBSET','CBL_POIDS','BLR_POIDS']]
        pairs['Mass Gap'] = pairs['CBL_POIDS']-pairs['BLR_POIDS']
        pairs = pairs.sort_values(by=['Mass Gap'])

        if cib.shape[0] == 0:
            continue

        plotframes = [cib, bel]
        ylim = style.ylim_no_outliers(plotframes)
        if (channel[2:6] == 'SIME') or (channel[2:6] == 'CVEH' and channel[14:15] == 'X'):
            ylim = (-60, 30)
        ylabel = style.ylabel(channel[12:14], channel[14:15])

        r, c = style.sqfactors(pairs.shape[0])
        fig, axs = style.subplots(r, c, sharex='all', sharey='all', visible=True,
                                  num='Mass: '+channel, figsize=(5*c, 3.125*r))
        fig.suptitle(
                '{sbdir} - Frontal Offset Crash Test at {speed} km/h\n{ch} - {desc}'.format(
                sbdir = subdir[:-4], speed = subdir[-3:-1], ch = channel,
                desc = description.get(channel, 'Description Unavailable')))

        for i, (cib_tcn, bel_tcn) in enumerate(zip(pairs.CIBLE, pairs.BELIER)):
            ax = axs[i]

            pair=pairs[pairs.CIBLE==cib_tcn]
            bool1 = pair.SUBSET.item() == 'HEV vs ICE'
            bool2 = not pair.SUBSET.item() == 'General'
            colors['cib'] = ('tab:blue' if bool1 else 'tab:purple') if bool2 else 'tab:pink'
            colors['bel'] = ('tab:orange' if bool1 else 'tab:green') if bool2 else 'tab:grey'

            ax.plot(time, cib.get(cib_tcn, default=time*np.nan), color=colors['cib'], label=cib_tcn)
            ax.plot(time, bel.get(bel_tcn, default=time*np.nan), color=colors['bel'], label=bel_tcn)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            title = '{} vs {}: {} vs {}\n({} lbs)'.format(cib_tcn, pair.BELIER.to_string(index=False),
                     pair.CBL_MODELE.to_string(index=False), pair.BLR_MODELE.to_string(index=False),
                     pair['Mass Gap'].to_string(index=False))
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.legend(loc=4)

        plt.tight_layout(rect=[0,0,1,0.92])
        plt.savefig(savedir+subdir+'Mass_'+channel+'.png', dpi=200)
        plt.close('all')