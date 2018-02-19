# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:33:47 2018

@author: giguerf
"""
import os
import numpy as np
import pandas as pd
from PMG.COM.openbook import openHDF5
from PMG.COM.outliers import check_and_clean, clean_outliers

def import_data(directory, channel, tcns=None, sl=slice(None), check=True):
    """Import a channel's dataframe, cleaned of outliers. Optionally filter by
    list of tcns and slice. Set check to False to skip 'view  outliers' prompt
    """
    time, fulldata = openHDF5(os.fspath(directory), [channel])
    data = fulldata[channel]
    raw = data[sl]
    t = time[sl]
    if tcns is not None:
        raw = raw.loc[:,tcns].dropna(axis=1)
    if check:
        clean = check_and_clean(raw)
    else:
        clean = clean_outliers(raw)
    return t, clean

def process(raw, norm=True, smooth=True, scale=True):
    """Preprocess data for use in calcuations, etc. set the appropriate flags
    for normalization, smoothings, and scaling
    """
    data = raw.copy()
    if norm:
        data = (data - data.mean())/data.std()
    if smooth:
        data = data.rolling(window=30, center=True, min_periods=0).mean()
        #TODO: data.ewm()
    if scale:
        sign = not positive_max(data)
        data = (data - data.min().min())/(data.max().max() - data.min().min())-int(sign)
    return data

def smooth(data):
    return process(data, norm=False, smooth=True, scale=False)

def scale(data):
    return process(data, norm=False, smooth=False, scale=True)

def positive_max(data):
    return bool(abs(data.max().max())>abs(data.min().min()))

def firstsmallest(arr):
    """Find the first value in the list that is not bigger than the rest by
    at least 10.
    """
    arr = np.array(arr)
    if len(arr)==1:
        return arr[0]
#    if arr[0]>arr[1]+10:
    if arr[0]>min(arr[1])+10:
        return firstsmallest(arr[1:])
    else:
        return arr[0]

def find_peaks(data, time=None, minor=False):
        return find_values(data, time, minor=minor, fun='min')
#    data = data.dropna(axis=1)
#    if time is None:
#        time = pd.Series(range(len(data)))
#    positive = positive_max(data) if (not minor) else (not positive_max(data))
#    peaks = data.max() if positive else data.min()
#    indices = data.idxmax() if positive else data.idxmin()
#    times = indices.apply(lambda i: time[int(i)])
#
#    return peaks, times

def find_values_old(data, time=None, scale=1, offset=0, minor=False):
        return find_values(data, time, scale, offset, minor, 'min')
#
#    data = data.dropna(axis=1)
#    if time is None:
#        time = pd.Series(range(len(data)))
#
#    positive = positive_max(data) if (not minor) else (not positive_max(data))
#    peaks = data.max() if positive else data.min()
#    values = peaks*scale+offset
#
#    stop = data.idxmax().max() if positive else data.idxmin().max()
#    diff = (data-values).iloc[:stop]
#    indices = diff.abs().idxmin()
#
#    times = indices.apply(lambda i: time[int(i)])
#
#    return values, times

def find_values(data, time=None, scale=1, offset=0, minor=False, fun=firstsmallest):
    """
    Input:
    ----------
    data : dataframe
        input data
    time : Series
        use for times output. otherwise returns index
    scale : float
        for 5% of peak use scale = 0.05
    offset : float
        for peak-10 use offset = -10
        for consistent value, use scale = 0 and offset = value
    minor : bool, default False
        Whether to use the max or min value will be determined by which
        absolute value is greatest. To use the opposite limit pass minor=True
    fun : 'min' or callable, default firstsmallest
        function used to find the index of the closest match
        pass min to evaluate the absolute closest match
        pass firstsmallest to find the closest small index
        pass callable that acts on a numpy array and returns the chosen value

    Returns:
    ----------
    values : Series
        values found in data
    times : Series
        time or index at which values appear

    Notes:
    ----------
        It does not return errors for values not in data.
        It searches only in the region before the peak value for each trace
    """
    dropped = pd.concat([data, data.dropna(axis=1)], axis=1).T.drop_duplicates(keep=False).T
    if not dropped.empty:
        data = data.dropna(axis=1)
        print('A column was dropped because it contained nan values. Please '
              'clean data before passing')
    if time is None:
        time = pd.Series(range(len(data)))
    if fun == 'min':
        fun = lambda x: x[0]

    positive = positive_max(data) if (not minor) else (not positive_max(data))
    peaks = data.max() if positive else data.min()
    values = peaks*scale+offset
    stop = data.idxmax() if positive else data.idxmin()
    diff = data.where(data.apply(lambda x: x.index<=stop[x.name]), other=np.inf)-values
    indices = np.argsort(diff.abs(), axis=0).apply(fun,0)
    times = indices.apply(lambda i: time[int(i)])

    return values, times

#def stats(df, old_way=False, band1=False, minmax=False):
#    import scipy.stats
#
#    def band(row, alpha, w=0):
#        """w = window (half-range), a = % band (0 to 1)"""
#        region = df.iloc[max(0,row.name-w):min(len(df),row.name+w)+1]
#        N = region.size
#        region = np.sort(np.array(region).reshape(N))
#        l = max(int(alpha*N), 0)
#        u = min(int((1-alpha)*N), int(N)-1)
#        return region[l], region[u]
#
#    def band2(row, alpha, gamma, w):
#        n=len(row)
#        um = int(scipy.stats.binom.ppf(1-alpha, p=gamma, n=n))
#        up = n+1 if (n-um)%2==1 else n
#        u, l = int((up+um)/2)-1, int((up-um)/2)-1
#        region = df.iloc[max(0,row.name-w):min(len(df),row.name+w)+1]
#        region = np.sort(np.array(region).reshape(region.size))
#        return region[l], region[u]
#
#    df = df.dropna(axis=1)
#    if old_way:
#        stats = {'Mean': df.mean(axis=1),
#                 'Low' : df.mean(axis=1)-1.96*df.std(axis=1),
#                 'High': df.mean(axis=1)+1.96*df.std(axis=1), #1.96 = 95% for normal distribution
#                 'Max' : df.max(axis=1),
#                 'Min' : df.min(axis=1)}
#    elif minmax:
#        stats = {'Mean': df.mean(axis=1),
#                 'Low' : df.min(axis=1),
#                 'High': df.max(axis=1)} #1.96 = 95% for normal distribution
#    else:
#        alpha = 0.05 #0.05 = 95% confidence
#        gamma = 0.90 #0.95 = 95% coverage
#        if band1:
#            highlow = df.apply(lambda row: band(row, alpha, w=0), axis=1)
#        else:
#            highlow = df.apply(lambda row: band2(row, alpha, gamma, w=0), axis=1)
#        highlow = list(zip(*highlow))
#        stats = {'Mean': df.mean(axis=1),
#                 'Low': pd.Series(highlow[0], index=df.index),
#                 'High' : pd.Series(highlow[1], index=df.index)}
#        over = df.T[(df.T>stats['High']*1.1).sum(axis=1)>30].T #20 = threshold for disqualification (spent more than 2.0 ms above 'High')
#        under = df.T[(df.T<stats['Low']*1.1).sum(axis=1)>30].T
#        between = pd.concat([df, over, under], axis=1).T.drop_duplicates(keep=False).T
#        stats['Mean-between'] = between.mean(axis=1)
#
#    stats = pd.DataFrame(data=stats)#.rolling(window=100, center=True, min_periods=0)
#
#    return stats

def stats(df):
    alpha = 0.05 #0.05 = 95% coverage

    df = df.dropna(axis=1)
    df2 = df.apply(lambda row: sorted(row), axis=1)
    df3 = df2.rolling(window=100, center=True, min_periods=0).mean()
    N = df.shape[1]

    stats = {'Mean': df.mean(axis=1),
             'Low': df3.iloc[:, np.ceil(N*alpha/2).astype(int)],
             'High' : df3.iloc[:, np.floor(N*(1-alpha/2)).astype(int)]}

    over_thresh = sorted((df.T>stats['High']*1).sum(axis=1))[N-N//10] #remove top 10% of data by time spent outside bounds
    under_thresh = sorted((df.T<stats['Low']*1).sum(axis=1))[N-N//10]
    over = df.T[(df.T>stats['High']*1).sum(axis=1)>=over_thresh].T
    under = df.T[(df.T<stats['Low']*1).sum(axis=1)>=under_thresh].T

    between = pd.concat([df, over, under], axis=1).T.drop_duplicates(keep=False).T
    stats['Mean-between'] = between.mean(axis=1)

    stats = pd.DataFrame(data=stats)#.rolling(window=100, center=True, min_periods=0)

    return stats
### 10ms rolling window of 95% data
#alpha = 0.05
#df2 = df.apply(lambda row: sorted(row), axis=1)
#N = df.shape[1]
#plt.figure()
#plt.plot(df.mean(axis=1))
#plt.plot(df2.rolling(window=100, center=True, min_periods=0).mean(), alpha=0.2)
#plt.plot(df2.iloc[:, int(N*alpha/2)].rolling(window=100, center=True, min_periods=0).mean())
#plt.plot(df2.iloc[:, int(N*(1-alpha/2))].rolling(window=100, center=True, min_periods=0).mean())
#plt.figure()
#plt.plot(df.mean(axis=1))
#plt.plot(df, alpha=0.2)
#plt.plot(df2.iloc[:, int(N*alpha/2)].rolling(window=100, center=True, min_periods=0).mean())
#plt.plot(df2.iloc[:, int(N*(1-alpha/2))].rolling(window=100, center=True, min_periods=0).mean())

#import matplotlib.pyplot as plt
#plt.close('all')
#values, times = find_values_old(data, time, scale=0, offset=-4.66, minor=False)
#values2, times2 = find_values(data, time, scale=0, offset=-4.66, minor=False)
#plt.plot(time, data)
#plt.plot(times, values,'.', markersize=8, label='1')
#plt.plot(times2, values2,'.', markersize=8, label='2')
#plt.legend()

#data = fulldata[chlist[1]].iloc[:,:10].rolling(window=100, center=True, min_periods=0).mean().dropna(axis=1)
#diff = data-(data.max()*0-5)
##idx_0 = diff.iloc[:,0].abs().argsort()[:5].min() #minimum of 5 closest matches
##diff_arr = np.array(diff)
##idxs = diff_arr.argsort(0)[diff_arr.argsort(0), np.arange(diff_arr.shape[1])][:5].min(0)
#indices = np.argsort(diff.abs(), axis=0)[:5].min()

#maxpeaks = pd.DataFrame()
#maxtimes = pd.DataFrame()
#for channel in chlist:
#    df = fulldata[channel].loc[:,tcns]
#    df = clean_outliers(df, 'data')
#    peaks, times = find_peaks(df, time)
#    maxpeaks[channel] = peaks
#    maxtimes[channel] = times
#
#plt.close('all')
#fig, axs = plt.subplots(4,4)
#axs = axs.flatten()
#for i, channel in enumerate(chlist):
#    ax = axs[i]
#    ax.plot(time, clean_outliers(fulldata[channel].loc[:,tcns], 'data'))
#    ax.plot(maxtimes[channel], maxpeaks[channel], '.')
#    ax.set_title(channel)
#
#rank = pd.DataFrame()
#direction = dict(zip(chlist, [False]*7+[True]*5+[False]*2+[True]*2))
#for channel in chlist:
#    df = fulldata[channel].loc[:,tcns]
#    sign = direction[channel]
#    rank[channel] = maxpeaks.sort_values(channel, axis=0, ascending=sign).index.tolist()
