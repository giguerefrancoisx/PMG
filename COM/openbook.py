# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:03:35 2017

@author: giguerf
"""
import os
import pandas
import PMG.COM.table as tb

def openbook(directory, channel=None):
    """
    USE OPENHDF5() WHEN POSSIBLE\n


    Reads all channel workbooks from the current directory into a single
    dictionary. Also creates a 'time' series
    """
    raise DeprecationWarning('All openbook functions not using HDF5 stores are no longer available.')
    if channel is not None:
        chdata = pandas.read_csv(directory+channel+'.csv')
        time = chdata.iloc[:4100,0]
        data = chdata.iloc[:,1:]
        return time, data

    fulldata = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            chdata = pandas.read_csv(directory+filename)
            fulldata[filename[:16]] = chdata.iloc[:,1:]

    time = chdata.iloc[:4100,0]

    return time, fulldata

def openHDF5(directory, channels=None):
    """
    Reads all channel workbooks from the current directory into a single
    dictionary. Also creates a 'time' series

    Input:
    ----------
    directory : str
        path to data storage.
    channels : str, default None
        optional, return data for a small list of channels from the set

    Returns:
    ----------
    time :
        series containing time channel.
    fulldata :
        dictionary containing a DataFrame for each channel.
    """
    try:
        with pandas.HDFStore(directory+'Channels.h5', mode='r+') as data_store:
            fulldata = {}

            if channels is not None:
                if isinstance(channels, str):
                    channels = [channels]
                for channel in channels:
                    chdata = data_store.select('C'+channel)
                    fulldata[channel] = chdata.iloc[:,1:]

            else:
                for key in data_store.keys():
                    chdata = data_store.select(key)
                    fulldata[key[2:]] = chdata.iloc[:,1:]

            time = chdata.iloc[:4100,0].round(4)
    except OSError:
        raise OSError('File Not Found. You are not conected to P://Projects '
                      'or you must first create the channels HDF5 store using '
                      'the writeHDF5 function\n')

    return time, fulldata

#import tables as tb
#tb.file._open_files.close_all()    #Close all open stores


def cibbel(readdir):
    """
    Reads all channel workbooks from the current directory into dictionaries
    separated by the selected population group. Also creates a 'time' series
    and returns pairs table
    """
    raise DeprecationWarning('All openbook functions not using HDF5 stores are no longer available.')
    cibdict = {}
    beldict = {}
    for filename in os.listdir(readdir):
        if filename.endswith('.csv'):

            chdata = pandas.read_csv(readdir+filename)
            pairs = lookup_pairs(chdata.columns[1:].tolist(), project='AHEC') #assign pairs via external function

            if chdata.shape[1] != 1:
                for tcn in pairs.BELIER.tolist()+pairs.CIBLE.tolist():
                    if tcn not in chdata.columns.tolist():
                        chdata[tcn] = [float('NaN') for i in range(chdata.shape[0])]

                cib = chdata[pairs.CIBLE]
                stats = {'Mean': cib.mean(axis = 1),
                         'High': cib.mean(axis = 1)+2*cib.std(axis = 1),
                         'Low': cib.mean(axis = 1)-2*cib.std(axis = 1)}
                cibstats = pandas.DataFrame(data = stats)
                cibstats = cibstats#.rolling(window=30,center=False).mean().shift(-15)
                cibdict[filename[:16]] = cib
                cibdict[filename[:16]+'_stats'] = cibstats

                bel = chdata[pairs.BELIER]
                stats = {'Mean': bel.mean(axis = 1),
                         'High': bel.mean(axis = 1)+2*bel.std(axis = 1),
                         'Low': bel.mean(axis = 1)-2*bel.std(axis = 1)}
                belstats = pandas.DataFrame(data = stats)
                belstats = belstats#.rolling(window=30,center=False).mean().shift(-15)
                beldict[filename[:16]] = bel
                beldict[filename[:16]+'_stats'] = belstats
            else:
                print('chdata  is blank for :', filename)
                cibdict[filename[:16]] = pandas.DataFrame()
                beldict[filename[:16]] = pandas.DataFrame()
                cibdict[filename[:16]+'_stats'] = pandas.DataFrame(columns = ['Mean', 'High', 'Low'])
                beldict[filename[:16]+'_stats'] = pandas.DataFrame(columns = ['Mean', 'High', 'Low'])

    time = chdata.iloc[:4100,0]
    time.name = 'Time'

    return time, cibdict, beldict, pairs

def thor(chlist, tcns=None):
    """Splits the data from the THOR project into slip and no slip tests."""
    THOR = 'P:/AHEC/DATA/THOR/'
    categories = ['SLIP','OK']
    column = 'CBL_BELT'
    return _categorize(THOR, 'THOR', chlist, tcns, column, categories)

def _categorize(path, project, chlist, tcns, column, categories):
    """Don't call this directly, instead call the desired function (eg: thor())

    Opens and splits the project data into the requested categories. See
    PMG.COM.table.get(), split() and this file's openHDF5() for parameter
    explanations.
    """
    table = tb.get(project)
    time, fulldata = openHDF5(path, chlist)
    split_table = tb.split(table, column, categories)
    category_dict = {}
    for category in categories:
        category_dict[str(category)] = {}
        table = split_table[category]
        if tcns is not None:
            table = table[table.CIBLE.isin(tcns)]
        category_tcns = table.CIBLE.tolist()

        for ch in chlist:
            chdata = fulldata[ch]
            df = chdata.loc[:, category_tcns].dropna(axis=1)
            category_dict[str(category)][ch] = df

    return time, category_dict

def gencut(readdir, group=''):
    """
    Reads all channel workbooks from the current directory into dictionaries
    separated by the selected population group. Also creates a 'time' series
    and table for each population group
    """
    raise DeprecationWarning('All openbook functions not using HDF5 stores are no longer available.')
    gendict = {}
    cutdict = {}
    for filename in os.listdir(readdir):
        if filename.endswith(".xlsx"):

            chdata = pandas.read_excel(readdir+'/'+filename)
            [genpop, cutpop] = lookup_pop(chdata.columns[1:].tolist(), group)

            gen = chdata[genpop.ALL]
            stats = {'Mean': gen.mean(axis = 1),
                     'High': gen.mean(axis = 1)+2*gen.std(axis = 1),
                     'Low': gen.mean(axis = 1)-2*gen.std(axis = 1)}
            genstats = pandas.DataFrame(data = stats)
            genstats = genstats#.rolling(window=30,center=False).mean().shift(-15)
            gendict[filename[:16]] = gen
            gendict[filename[:16]+'_stats'] = genstats

            cut = chdata[cutpop.ALL]
            stats = {'Mean': cut.mean(axis = 1),
                     'High': cut.mean(axis = 1)+2*cut.std(axis = 1),
                     'Low': cut.mean(axis = 1)-2*cut.std(axis = 1)}
            cutstats = pandas.DataFrame(data = stats)
            cutstats = cutstats#.rolling(window=30,center=False).mean().shift(-15)
            cutdict[filename[:16]] = cut
            cutdict[filename[:16]+'_stats'] = cutstats

    time = chdata.iloc[:4100,0]
    time.name = 'Time'

    return time, gendict, cutdict, genpop, cutpop

def oldnew(readdir):
    """
    Reads all channel workbooks from the current directory into dictionaries
    separated by 'old' and 'new'. Similar to cib/bel for AHEC.
    """
    raise DeprecationWarning('All openbook functions not using HDF5 stores are no longer available.')
    olddict = {}
    newdict = {}
    missing = []
    for filename in os.listdir(readdir):
        if filename.endswith(".xlsx"):

            chdata = pandas.read_excel(readdir+'/'+filename)
            pairs = lookup_pairs(chdata.columns[1:].tolist(), 'sled')

            if chdata.shape[1] != 1:
                for tcn in pairs.NEW.tolist()+pairs.OLD.tolist():
                    if tcn not in chdata.columns.tolist():
                        missing.append(tcn)
                        chdata[tcn] = [float('NaN') for i in range(chdata.shape[0])] ###drop duplicates?

                old = chdata[pairs.OLD]
                stats = {'Mean': old.mean(axis = 1),
                         'High': old.mean(axis = 1)+2*old.std(axis = 1),
                         'Low': old.mean(axis = 1)-2*old.std(axis = 1)}
                oldstats = pandas.DataFrame(data = stats)
                oldstats = oldstats#.rolling(window=30,center=False).mean().shift(-15)
                olddict[filename[:16]] = old
                olddict[filename[:16]+'_stats'] = oldstats

                new = chdata[pairs.NEW]
                stats = {'Mean': new.mean(axis = 1),
                         'High': new.mean(axis = 1)+2*new.std(axis = 1),
                         'Low': new.mean(axis = 1)-2*new.std(axis = 1)}
                newstats = pandas.DataFrame(data = stats)
                newstats = newstats#.rolling(window=30,center=False).mean().shift(-15)
                newdict[filename[:16]] = new
                newdict[filename[:16]+'_stats'] = newstats

    return olddict, newdict, pairs

def populations(readdir, table, column='SUBSET'):
    """
    Must have columns Cible and Belier
    Must pass column with the subset types.
    """
    raise DeprecationWarning('All openbook functions not using HDF5 stores are no longer available.')
    popdict = {}
    missing = []
    for filename in os.listdir(readdir):
        if filename.endswith('.csv'):
            channel = filename[:16]
            popdict[channel] = {}
            popdict[channel+'_stats'] = {}
            chdata = pandas.read_csv(readdir+filename)

            if chdata.shape[1] != 1:
                for tcn in table.CIBLE.tolist()+table.BELIER.tolist():
                    if tcn not in chdata.columns.tolist():
                        missing.append(tcn)
                        chdata[tcn] = [float('NaN') for i in range(chdata.shape[0])]

                for subset in set(table[column]):
                    popdict[channel][subset] = {}
                    popdict[channel+'_stats'][subset] = {}

                    for pos in ['CIBLE','BELIER']:
                        pop = chdata[table[table[column]==subset].loc[:,pos]].dropna(axis=1)
                        stats = {'Mean': pop.mean(axis = 1),
                                 'High': pop.mean(axis = 1)+2*pop.std(axis = 1),
                                 'Low': pop.mean(axis = 1)-2*pop.std(axis = 1)}
                        popstats = pandas.DataFrame(data = stats)
                        popdict[channel][subset][pos] = pop
                        popdict[channel+'_stats'][subset][pos] = popstats

    return popdict

def grids(fulldata, chlist):
    """
    Separates the list of tests by speed and collision type (grids). Returns
    a dictionary of these grids by group.
    """
    raise DeprecationWarning('All openbook functions not using HDF5 stores are no longer available.')
    groupdict = {}
    for group in ['A', 'B', 'C', 'D', 'E','']:
        groupdict[group] = {}
        for channel in chlist:
            groupdict[group][channel] = {}

            chframe = fulldata[channel]
            [P, subset] = lookup_pop(chframe.columns[1:].tolist(),group)
            subset = (P if group == '' else subset)
            offset = subset[subset['Type'] == 'Frontale/Véhicule']
            mur = subset[subset['Type'] == 'Frontale/Mur']

            off48 = chframe[offset[offset['Vitesse'] == 48].ALL]
            off56 = chframe[offset[offset['Vitesse'] == 56].ALL]
            mur48 = chframe[mur[mur['Vitesse'] == 48].ALL]
            mur56 = chframe[mur[mur['Vitesse'] == 56].ALL]

            gridtypes = ['off48','mur48','off56','mur56']
            groupdict[group][channel] = dict(zip(gridtypes, [off48,mur48,off56,mur56]))

    return groupdict

def popgrids(fulldata, chlist):
    """
    Separates the list of tests by speed and collision type (grids). Returns
    a dictionary of these grids by group.
    """
    raise DeprecationWarning('All openbook functions not using HDF5 stores are no longer available.')
    groupdict = {}
    for group in ['A','B','C','UAS','']:
        groupdict[group] = {}
        for channel in chlist:
            groupdict[group][channel] = {}

            chframe = fulldata[channel]
            [P, subset] = lookup_pop(chframe.columns[1:].tolist(),group)
            subset = (P if group == '' else subset)
            offs = subset[subset['Type'] == 'Frontale/Véhicule']
            murs = subset[subset['Type'] == 'Frontale/Mur']
            pris = subset[subset['Type'] == 'Prius Body']

            off = chframe[offs.ALL]
            mur = chframe[murs.ALL]
            pri = chframe[pris.ALL]

            gridtypes = ['offset','mur','prius']
            groupdict[group][channel] = dict(zip(gridtypes, [off,mur,pri]))

    return groupdict
#%%

#veh = veh[veh.index.isin(TCNs)]
#sled = sled[sled.index.isin(TCNs)]

def lookup_pop(TCNs, group):
    """
    Returns two DataFrames with information gathered from the boostertable,
    separated by population group.
    """
    raise DeprecationWarning('All openbook functions not using HDF5 stores are no longer available.')
    table = pandas.read_excel('P:/BOOSTER/boostertable.xlsx')
    table = table[table.loc[:,['TC_Vehicule']].isin(TCNs).any(axis=1)]
    population = table.loc[:,['TC_Vehicule', 'Type', 'Vitesse',
                              'Marque', 'Modele', 'Group']]
    population = population.set_index('TC_Vehicule')

    subset = population[population['Group'] == group]
    subset = subset.drop_duplicates()
    population = population.drop(subset.index)
    population = population.dropna(axis = 0, thresh = 2)

    return population, subset

def lookup_pairs(TCNs=None, project=None):
    """
    Returns a DataFrame with information gathered from the project's table.
    Automatically pairs the TCNs selected.
    """
    raise DeprecationWarning('All openbook functions not using HDF5 stores are no longer available.')
    if project == 'sled':
        table = pandas.read_excel('P:/SLED/sledtable.xlsx')
        table = table.dropna(axis=0, how='all').dropna(axis=1, thresh=2)
        if TCNs is not None:
            table = table[table.loc[:,['OLD','NEW']].isin(TCNs).any(axis=1)]
        pairs = table.loc[:,['OLD','NEW','GROUP']]

    elif project == 'AHEC':
        table = pandas.read_excel('P:/AHEC/ahectable.xlsx')
        table = table.dropna(axis=0, how='all').dropna(axis=1, thresh=2)
        if TCNs is not None:
            table = table[table.loc[:,['CIBLE','BELIER']].isin(TCNs).any(axis=1)]
        pairs = table.loc[:,['CIBLE','BELIER','VITESSE','CBL_MODELE','BLR_MODELE','SUBSET']]

    elif project == 'THOR':

        singles = pandas.read_excel('P:/AHEC/thortable.xlsx', sheetname='All')
        singles = singles.dropna(axis=0, thresh=5).dropna(axis=1, how='all')
        pairs = pandas.read_excel('P:/AHEC/thortable.xlsx', sheetname='Pairs')
        pairs = pairs.dropna(axis=0, thresh=5).dropna(axis=1, how='all')

        if TCNs is not None:
            singles = singles[singles.loc[:,'CIBLE'].isin(TCNs)]
            pairs = pairs[pairs.loc[:,['CIBLE','BELIER']].isin(TCNs).any(axis=1)]

        return singles, pairs

    else:
        print('need to add code for project {}'.format(project))

    return pairs

