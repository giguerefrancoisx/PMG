# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:44:02 2018

@author: giguerf
"""
import pandas as pd
import matplotlib.pyplot as plt

#def get_limits(data):
#    """Returns the ylimits necessary to correctly plot the dataset passed as
#    input. You may pass a dataframe or list of dataframes to evaluate.
#    """
#
#    if isinstance(data, list):
#        data = pd.concat(data, axis=1)
#    else:
#        data = data
#    data = data.T[data.any()].T
#
#    maxlist = data.max()
#    maxlist.name = 'max'
#    maxlist = maxlist.sort_values(ascending=False).reset_index()
#    above_average = maxlist.loc[0,'max'] > 6*maxlist.loc[1:,'max'].mean()
#    significant = maxlist.loc[0,'max'] > maxlist.loc[1,'max']+1
#    if above_average and significant:
#        tcn = maxlist.loc[0,'index']
#        ymax = maxlist.loc[1,'max']
#        ymin, ymax = get_limits(data.drop(tcn, axis=1))
#    else:
#        ymax = maxlist.loc[0,'max']
#
#    minlist = data.min()
#    minlist.name = 'min'
#    minlist = minlist.sort_values(ascending=True).reset_index()
#    above_average = minlist.loc[0,'min'] < 6*minlist.loc[1:,'min'].mean()
#    significant = minlist.loc[0,'min'] < minlist.loc[1,'min']-1
#    if above_average and significant:
#        tcn = minlist.loc[0,'index']
#        ymin = minlist.loc[1,'min']
#        ymin, ymax = get_limits(data.drop(tcn, axis=1))
#    else:
#        ymin = minlist.loc[0,'min']
#
#    return ymin, ymax
#
#def drop_outliers(data):
#    """Returns the data without outliers. You may pass a dataframe or list of
#    dataframes to evaluate.
#    """
#
#    if isinstance(data, list):
#        data = pd.concat(data, axis=1)
#    else:
#        data = data
#    data = data.T[data.any()].T
#
#    maxlist = data.max()
#    maxlist.name = 'max'
#    maxlist = maxlist.sort_values(ascending=False).reset_index()
#    above_average = maxlist.loc[0,'max'] > 6*maxlist.loc[1:,'max'].mean()
#    significant = maxlist.loc[0,'max'] > maxlist.loc[1,'max']+1
#    if above_average and significant:
#        tcn = maxlist.loc[0,'index']
#        data = drop_outliers(data.drop(tcn, axis=1))
#
#    minlist = data.min()
#    minlist.name = 'min'
#    minlist = minlist.sort_values(ascending=True).reset_index()
#    above_average = minlist.loc[0,'min'] < 6*minlist.loc[1:,'min'].mean()
#    significant = minlist.loc[0,'min'] < minlist.loc[1,'min']-1
#    if above_average and significant:
#        tcn = minlist.loc[0,'index']
#        data = drop_outliers(data.drop(tcn, axis=1))
#
#    return data

def clean_outliers(data, returning):
    """Returns either the input data after dropping outliers, or the minimum
    and maximum of that data.

    Inputs:
        data: You may pass a dataframe or list of dataframes to evaluate.

        returning: either 'data' or 'limits'
    """
    output = None
    if isinstance(data, list):
        data = pd.concat(data, axis=1)
    data = data.T[data.any()].T

    maxlist = data.max()
    maxlist.name = 'max'
    maxlist = maxlist.sort_values(ascending=False).reset_index()
    above_average = maxlist.loc[0,'max'] > 6*maxlist.loc[1:,'max'].mean()
    significant = maxlist.loc[0,'max'] > maxlist.loc[1,'max']+1
    if above_average and significant:
        tcn = maxlist.loc[0,'index']
        ymax = maxlist.loc[1,'max']
        output = clean_outliers(data.drop(tcn, axis=1), returning)
    else:
        ymax = maxlist.loc[0,'max']

    minlist = data.min()
    minlist.name = 'min'
    minlist = minlist.sort_values(ascending=True).reset_index()
    above_average = minlist.loc[0,'min'] < 6*minlist.loc[1:,'min'].mean()
    significant = minlist.loc[0,'min'] < minlist.loc[1,'min']-1
    if above_average and significant:
        tcn = minlist.loc[0,'index']
        ymin = minlist.loc[1,'min']
        output = clean_outliers(data.drop(tcn, axis=1), returning)
    else:
        ymin = minlist.loc[0,'min']

    if returning == 'limits':
        if output is not None:
            ymin, ymax = output
        return ymin, ymax
    if returning == 'data':
        if output is not None:
            data = output
        return data

def check_and_clean(raw):
    """Check thats the data does not contain outliers as found by
    clean_outliers(). Prompts the user for whether a plot highlighting the
    outliers should be shown. Options are to retain or discard the outliers.
    """
    raw = raw.T[raw.any()].T #filter out bad/missing traces
    clean = clean_outliers(raw, 'data')
    outliers = pd.concat([raw, clean], axis=1).T.drop_duplicates(keep=False).T

    if not outliers.empty:
        action = input("Outliers have been detected.\nview: 'y' \ndiscard: any\nretain: 'keep'\n>>>")
        if action == 'y':
            plt.figure()
            plt.plot(clean, alpha=0.25)
            plt.plot(outliers)
            plt.waitforbuttonpress()
            plt.close('all')
            action = input("discard: any\nretain: 'keep'\n>>>")

        if action == 'keep':
            clean = raw
        else:
            print('outlier(s) ignored: '+', '.join(outliers.columns.tolist())+'\n')

    return clean

#ymin, ymax = get_limits(raw)
#df = drop_outliers(raw)
#out = clean_outliers(data, 'limits')