# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:33:47 2018

@author: giguerf
"""
import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from PMG.COM.openbook import openHDF5

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

def process(raw, norm=True, smooth=True, scale=True, win_type='parzen'):
    """Preprocess data for use in calcuations, etc. set the appropriate flags
    for normalization, smoothings, and scaling. The win_type should be one of:
    'boxcar', 'triang, 'parzen'. boxcar procudes a simple moving average,
    triang produces a linearly weigthed average, and parzen procudes a 4th
    order B-spline window weighted average.
    """
    data = raw.copy()
    if norm:
        data = (data - data.mean())/data.std()
    if smooth:
        data = data.rolling(window=30, center=True, min_periods=0, win_type=win_type).mean()
    if scale:
        sign = not positive_max(data)
        data = (data - data.min().min())/(data.max().max() - data.min().min())-int(sign)
    return data

def smooth(data, win_type):
    return process(data, norm=False, smooth=True, scale=False, win_type=win_type)

def scale(data):
    return process(data, norm=False, smooth=False, scale=True)

def positive_max(data):
    """if the absolute largest value is positive, positive_max = True"""
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
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

def find_peak(data, time=None, minor=False):
        return find_value(data, time, minor=minor, fun='min')

def find_value_old(data, time=None, scale=1, offset=0, minor=False):
        return find_value(data, time, scale, offset, minor, 'min')

def find_value(data, time=None, scale=1, offset=0, minor=False, fun=firstsmallest):
    """
    Input:
    ----------
    data : Series or DataFrame
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
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
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

def find_all_peaks(array, minor=False):
    """Find all the minima, maxima, or both on an array."""
    positive = positive_max(array) if (not minor) else (not positive_max(array))

    if not positive: #find minimum values
        pts = (np.diff(np.sign(np.diff(array))) > 0).nonzero()[0] + 1
    elif positive: #find maximum values
        pts = (np.diff(np.sign(np.diff(array))) < 0).nonzero()[0] + 1
    else:
        pts = np.diff(np.sign(np.diff(array))).nonzero()[0] + 1
    return pts

def bounds(df, tcn):
    """return the left, right, upper, and lower bounds of a peak"""
    column = df[tcn]
    lower, upper = df.min().min(), df.max().max()
    positive = abs(upper)>abs(lower)
    through_zero = (upper>=0) and (lower<=0)
    if through_zero:
        lower, upper = (-lower, upper) if positive else (lower, -upper)
    else:
        mean = df.mean().mean()
        lower, upper = (mean+lower, upper) if positive else (lower, mean-upper)
    limit = lower if positive else upper

    search_range = np.nonzero(column/limit>1)
    left, right = np.min(search_range), np.max(search_range)

    return left, right, lower, upper

def smooth_peaks(data):
    """Find the least-smoothed curve such that there exists a single peak in
    the range of interest.

    Input:
    ----------
    data : Series or Dataframe
        input data
    return_windows : bool (deactivated for now)
        flag to set to output windows used for smoothing

    Returns:
    ----------
    smooth_data : Series or DataFrame, like given
        values found in data

    >>> smooth_data = smooth_peaks(data)
    >>> peaks, times = find_peaks(smooth_data, time)

    Notes:
    ----------
    Peaks found are garanteed to be lesser in magnitude than the raw data. The
    purpose of this function is to find an appropriate time-to-peak value for
    oscilating data.
    """
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
        return_series=True
    else:
        return_series=False
    dropped = pd.concat([data, data.dropna(axis=1)], axis=1).T.drop_duplicates(keep=False).T
    if not dropped.empty:
        data = data.dropna(axis=1)
        print('A column was dropped because it contained nan values. Please '
              'clean data before passing')

    smooth_data = pd.DataFrame(columns=data.columns)

    for tcn in data.columns:
        N=0
        smooth=False
        left, right, lower, upper = bounds(data, tcn)
        while not smooth:
            N+=10
            smoothed = data[tcn].rolling(window=N, center=True, min_periods=0, win_type='triang').mean()
            pts = find_all_peaks(smoothed)
            # if there exists a single peak within the bounds, it is now smooth
            if len(pts[(lower <= data[tcn].loc[pts].values) & \
                       (data[tcn].loc[pts].values <= upper) & \
                       (left <= pts) & (pts <= right)]) == 1:
                smooth = True
            if N>500:
                break
        smooth_data[tcn] = smoothed

    if return_series:
        return smooth_data[tcn]
    else:
        return smooth_data

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

def stats(df):
    alpha = 0.05 #0.05 = 95% coverage

    df = df.dropna(axis=1)
    df2 = df.apply(lambda row: sorted(row), axis=1)
    df3 = df2.rolling(window=100, center=True, min_periods=0, win_type='boxcar').mean()
    N = df.shape[1]

    stats = {'Mean': df.mean(axis=1),
             'Low': df3.iloc[:, np.ceil(N*alpha/2).astype(int)],
             'High' : df3.iloc[:, np.floor(N*(1-alpha/2)).astype(int)-1]}

    over_thresh = sorted((df.T>stats['High']*1).sum(axis=1))[N-N//10] #remove top 10% of data by time spent outside bounds
    under_thresh = sorted((df.T<stats['Low']*1).sum(axis=1))[N-N//10]
    over = df.T[(df.T>stats['High']*1).sum(axis=1)>=over_thresh].T
    under = df.T[(df.T<stats['Low']*1).sum(axis=1)>=under_thresh].T

    between = pd.concat([df, over, under], axis=1).T.drop_duplicates(keep=False).T
    stats['Mean-between'] = between.mean(axis=1)

    stats = pd.DataFrame(data=stats)

    return stats

def average_rows(df, window):
    sampled_df = df[::window].copy()
    for l, r in zip(sampled_df.index, sampled_df.index+window):
        sampled_df.loc[l,:] = df.loc[l:r,:].mean(axis=0)
    return sampled_df

# TODO - upsample back to full size?

def stats_kde(df):
    """Estimate cdf based on kde. Use alpha to determine upper and lower
    limits on data"""

    def kde_band(row):
        alpha = 3/(len(row)+3) #heuristic lower bound for level of confidence
        kde = scipy.stats.gaussian_kde(row, bw_method=0.25) #bandwidth more aggressive that default
        x = np.linspace(min(row)*1.5, max(row)*1.5, 1000)
        cdf = np.cumsum(kde(x))/np.sum(kde(x))
        lower = x[np.argmin(np.abs(cdf-alpha/2))]
        upper = x[np.argmin(np.abs(cdf-(1-alpha/2)))]
        return lower, upper

    df = clean_outliers(df.dropna(axis=1), 'data')
    N = df.shape[1]
    df2 = average_rows(df, 5)
    df2 = df2.apply(lambda row: kde_band(row), axis=1)
    stats = pd.DataFrame(df2.tolist(), index=df2.index)
    stats.columns=['Low','High']
    stats['Mean'] = df.mean(axis=1)

    over_thresh = sorted((df.T>stats['High']*1).sum(axis=1))[N-N//10] #remove top 10% of data by time spent outside bounds
    under_thresh = sorted((df.T<stats['Low']*1).sum(axis=1))[N-N//10]
    over = df.T[(df.T>stats['High']*1).sum(axis=1)>=over_thresh].T
    under = df.T[(df.T<stats['Low']*1).sum(axis=1)>=under_thresh].T

    between = pd.concat([df, over, under], axis=1).T.drop_duplicates(keep=False).T
    stats['Mean-between'] = between.mean(axis=1)
    N_between = between.shape[1]

    stats = stats.rolling(window=50, center=True, min_periods=0, win_type='parzen').mean()

    return stats, N_between, 3/(N+3)

### kde-based cdf
#plt.figure()
#row = df.iloc[1600]
#row.hist(density=True)
#x = np.linspace(min(row)-2, max(row)+3, 100)
#kde = scipy.stats.gaussian_kde(row)
#pdf = kde(x)
#cdf = np.cumsum(kde(x))/np.sum(kde(x))
#plt.plot(x, pdf)
#plt.plot(x, cdf)
#ax=plt.gca()
#alpha=0.05
#ax.axvline(x[np.argmin(np.abs(cdf-alpha/2))])
#ax.axvline(x[np.argmin(np.abs(cdf-(1-alpha/2)))])

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

#import sklearn.mixture
#import scipy.stats
#def band(row):
#    #row = df.iloc[333]
#    alpha = 0.05
#    row2 = np.array(row).reshape(-1,1)
#    gm = sklearn.mixture.GaussianMixture(2)
#    gm.fit(row2)
#    fit1 = gm.weights_[0],gm.means_[0][0],gm.covariances_[0][0][0]
#    fit2 = gm.weights_[1],gm.means_[1][0],gm.covariances_[1][0][0]
#    x = np.linspace(row.min(),row.max(),1000)
#    y1 = scipy.stats.norm.pdf(x, loc=fit1[1], scale=fit1[2])
#    y2 = scipy.stats.norm.pdf(x, loc=fit2[1], scale=fit2[2])
#    y = fit1[0]*y1+fit2[0]*y2
#    cdf = np.cumsum(y)/np.cumsum(y)[-1]
#    lower = x[np.argmin(np.abs(cdf-alpha))]
#    mean = x[np.argmin(np.abs(cdf-0.5))]
#    upper = x[np.argmin(np.abs(cdf-(1-alpha)))]
#    return lower, mean, upper
#
#def stats2(df):
##    alpha = 0.05 #0.05 = 95% coverage
#
#    df = df.dropna(axis=1)
#    res = df.apply(lambda row: band(row), axis=1)
#    stats = pd.DataFrame(np.array(list((zip(*res)))).T, columns = ['Low', 'Mean', 'High'])
#    N = df.shape[1]
#
##    stats = {'Mean': df.mean(axis=1),
##             'Low': df3.iloc[:, np.ceil(N*alpha/2).astype(int)],
##             'High' : df3.iloc[:, np.floor(N*(1-alpha/2)).astype(int)-1]}
#
#    over_thresh = sorted((df.T>stats['High']*1).sum(axis=1))[N-N//10] #remove top 10% of data by time spent outside bounds
#    under_thresh = sorted((df.T<stats['Low']*1).sum(axis=1))[N-N//10]
#    over = df.T[(df.T>stats['High']*1).sum(axis=1)>=over_thresh].T
#    under = df.T[(df.T<stats['Low']*1).sum(axis=1)>=under_thresh].T
#
#    between = pd.concat([df, over, under], axis=1).T.drop_duplicates(keep=False).T
#    stats['Mean-between'] = between.mean(axis=1)
#
#    stats = pd.DataFrame(data=stats)#.rolling(window=100, center=True, min_periods=0)
#
#    return stats

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
