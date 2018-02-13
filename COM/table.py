# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:42:05 2018

@author: giguerf
"""
import pandas as pd

def get(project):
    """Given a project name (eg: 'THOR') return the appropriate table"""
    if project == 'THOR':
        table = pd.read_excel('P:/AHEC/thortable.xlsx', sheetname='All')
    if project =='AHEC':
        table = pd.read_excel('P:/AHEC/ahectable.xlsx')
    if project == 'BOOSTER':
        table = pd.read_excel('P:/BOOSTER/boostertable.xlsx')

    table = table.dropna(axis=0, thresh=5).dropna(axis=1, how='all')
    return table

def split(table, column, categories=None):
    """Split the table into a dictionary of tables, with the keys being the
    categories passed.

    Input:
    ------------
    table : DataFrame
        Table to be split.
    column : str
        Name of the column to search for values matching the categories.
    categories : list, default None
        List of keys to split table by. If None, use the set of unique values
        from that column

    Returns:
    -------------
    tables : dict
        Dictionary of tables corresponding to each category

    Example:
    -----------
    >>> tables = split(table, 'VITESSE', [48, 56])
    >>> {48 : DataFrame, 56 : DataFrame}
    """
    if categories is None:
        categories = set(table[column])

    tables = {}
    for cat in categories:
        tables[cat] = table[table[column].isin([cat])]

    return tables

def tcns(tables, column='CIBLE', return_dict=False):
    """Return tcns or other values from a column on each table passed.

    Input:
    ------------
    tables : DataFrame, list, or dict
        Tables to extract values from
    column : str, default 'CIBLE'
        Name of the column of interest.
    return_dict : bool, default False
        Return the set of tcn lists as a dictionary instead. Useful when the
        tables are in a dictionary which was not keyed by hand. If used on a
        list of tables, keys will be the position in the list.

    Returns:
    -------------
    tcns_list : list or dict
        List or dictionary of tcns/values
    """

    if isinstance(tables, pd.core.frame.DataFrame):
        if return_dict:
            return {0: tables[column].tolist()}
        else:
            return [tables[column].tolist()]

    elif isinstance(tables, list):
        if return_dict:
            tcns_dict = {}
            for key, table in enumerate(tables):
                tcns_dict[key] = table[column].tolist()
            return tcns_dict
        else:
            tcns_list = []
            for table in tables:
                tcns_list.append(table[column].tolist())
            return tcns_list

    elif isinstance(tables, dict):
        if return_dict:
            tcns_dict = {}
            for key, table in tables.items():
                tcns_dict[key] = table[column].tolist()
            return tcns_dict
        else:
            tcns_list = []
            for table in tables.values():
                tcns_list.append(table[column].tolist())
            return tcns_list

def recursive_split(tables, columns, categories):
    """Split the table into a dictionary of tables, with the keys being the
    cominations of categories passed.

    Input:
    ------------
    tables : DataFrame or dict
        Tables to be split.
    column : list
        Names of the columns to search for values matching the categories.
    categories : list of lists
        List of keys to split table by.

    Returns:
    -------------
    tables : dict
        Dictionary of tables corresponding to each category

    Example:
    -----------
    >>> tables = recursive_split(table, ['VITESSE','CBL_BELT'], [[48, 56],['OK','SLIP']])
    >>> {48/OK : DataFrame, 56/OK : DataFrame, 48/SLIP : DataFrame, 56/SLIP : DataFrame}
    """
    for column, cats in zip(columns, categories):
        tables_temp = {}
        if isinstance(tables, dict):
            for key, table in tables.items():
                temp = split(table, column, cats)
                for k, v in temp.items():
                    tables_temp[str(key)+'/'+str(k)] = v
            tables = tables_temp
        else:
            tables = split(tables, column, cats)

    return tables

#if __name__ == '__main__':
#
#    table = pd.read_excel('P:/AHEC/thortable.xlsx', sheetname='All')
#    table = table.dropna(axis=0, thresh=5).dropna(axis=1, how='all')
#    out1 = split(table, 'CBL_BELT', categories=None)
#    out2 = tcns(out1, column='CIBLE', return_dict=True)
#
#    out3 = recursive_split(table, columns=['TYPE','CBL_BELT','VITESSE'], categories=[['Frontale/Mur','Frontale/Véhicule'],['SLIP','SLIDE','OK'],[48, 56]])
#    out4 = tcns(out3, return_dict=True)
