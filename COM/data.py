# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:33:47 2018

@author: giguerf
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PMG.COM.openbook as ob

def import_data(directory, channels, tcns=None, sl=slice(None), check=True, stage=1):
    """Import a channel's or channels' dataframe(s), cleaned of outliers. Optionally filter by
    list of tcns and slice. Set check to False to skip 'view  outliers' prompt

    Input:
    ----------
    directory : str
        path to HDF5 store
    channels : str or list
        channels for which to return data. If string, returns DataFrame rather than dictionary
    tcns : list
        list of tcns to include. tcns missing data will automatically be dropped
    sl : slice object
        slice for rows.
    check : bool
        If false, bypass checking and drop any outliers found by clean_outliers()

    Returns:
    ----------
    time : Series
        corresponding time channel
    clean : dict (or DataFrame)
        cleaned data ready for use.
    """
    return_single = False
    if isinstance(channels, str):
        channels = [channels]
        return_single = True
    time, fulldata = ob.openHDF5(os.fspath(directory), channels)
    t = time[sl]

    clean = {}
    for channel in channels:
        raw = fulldata[channel][sl]

        if tcns is not None:
            raw = raw.loc[:,tcns].dropna(axis=1)
        if check:
            clean[channel] = check_and_clean(raw, stage)
        else:
            clean[channel] = clean_outliers(raw, stage)
    if return_single:
        return t, clean[channel]
    else:
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
        data = data.rolling(30, 0, center=True, win_type=win_type).mean()
    if scale:
        sign = not positive_max(data)
        data = (data-data.min().min())/(data.max().max()-data.min().min())-int(sign)
    return data

def smooth(data, win_type='parzen'):
    return process(data, norm=False, smooth=True, scale=False, win_type=win_type)

def scale(data):
    return process(data, norm=False, smooth=False, scale=True)

def positive_max(data):
    """if the absolute largest median value is positive, positive_max = True"""
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    return bool(abs(data.max().median())>abs(data.min().median()))

def firstsmallest(arr):
    """Find the first value in the list that is not bigger than the rest by
    at least 10.
    """
    arr = np.array(arr)
    if len(arr)==1:
        return arr[0]
    if arr[0]>arr[1]+10:
#    if arr[0]>min(arr[1])+10:
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
        fun = lambda x: x.iloc[0]

    positive = positive_max(data) if (not minor) else (not positive_max(data))
    peaks = data.max() if positive else data.min()
    values = peaks*scale+offset
    stop = data.idxmax() if positive else data.idxmin()
    diff = data.where(data.apply(lambda x: x.index<=stop[x.name]), other=np.inf)-values
    indices = np.argsort(diff.abs(), axis=0).apply(fun,0)
    times = indices.apply(lambda i: time[time.index[0]+int(i)])

    return values, times

def find_all_peaks(array, minor=False, override=None):
    """Find all the minima, maxima, or both on an array."""
    positive = positive_max(array) if (not minor) else (not positive_max(array))
    if override is not None and override in ['min', 'max']:
        positive = True if override == 'max' else False

    if not positive: #find minimum values
        pts = (np.diff(np.sign(np.diff(array))) > 0).nonzero()[0] + 1
    elif positive: #find maximum values
        pts = (np.diff(np.sign(np.diff(array))) < 0).nonzero()[0] + 1
    else:
        pts = np.diff(np.sign(np.diff(array))).nonzero()[0] + 1
    return pts

def bounds(df): #TODO figure out a better lower, upper calculation
    """return the left, right, upper, and lower bounds of a peak"""
    lower, upper = df.min().min(), df.max().max()
    positive = abs(df.max().median())>abs(df.min().median())
    through_zero = df.max().median()*df.min().median()<0
    if through_zero:
        lower, upper = (df.max().min()/2, df.max().max()) if positive else (df.min().min(), df.min().max()/2)
    else:
        raise Exception('This case isn\'t coded yet')
#        mean = df.mean().mean() #Use Median?
#        lower, upper = (mean+lower, upper) if positive else (lower, mean-upper)

    return lower, upper

def smooth_peaks(data, time=None, win_incr=20, override=None, bounds=None, thresh=1):
    """Find the least-smoothed curve such that there exists a single peak in
    the range of interest.

    Input:
    ----------
    data : Series or Dataframe
        input data
    time : Series, optional
        Corresponding time
    win_incr : int, default 20
        Increment for smoothing window
    override : 'min' or 'max'
        Type of peak to find. If none, calls positive_max(data)
    bounds : (left, right, lower, upper) or 'limits'
        bounding box to search for peaks inside of. If time is not passed,
        index of time series must be used for left, right. Pass 'limits' to
        default to data limits (time[0] to time[-1], data min to data max).
        If None, find_bounds will automatically choose good-ish bounds for
        each column
    thresh : int, default 1
        maximum number of peaks inside

    Returns:
    ----------
    smooth_data : Series or DataFrame
        output smoothed data with single peak within bounds

    >>> smooth_data = smooth_peaks(data)
    >>> peaks, times = find_peaks(smooth_data, time)

    Notes:
    ----------
    Peaks found are garanteed to be lesser in magnitude than the raw data. The
    purpose of this function is to find an appropriate time-to-peak value for
    oscillating data.
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
    if time is None:
        time = pd.Series(np.arange(data.shape[0]))
    if override is None:
        override = 'max' if positive_max(data) else 'min'

    if (bounds is not None) and (bounds != 'limits'):
        left, right, lower, upper = bounds
    elif bounds == 'limits':
        lower, upper = data.min().min(), data.max().max()
        left, right = time.iloc[0], time.iloc[-1]

    smooth_data = pd.DataFrame(columns=data.columns)

    for tcn in data.columns:
        N=0
        smooth=False

        if bounds is None: #find automatically
#            left, right, lower, upper = find_bounds(data)
            raise NotImplementedError("Neeed to code 'bounds' finder")

        while not smooth:
            N+=win_incr
            smoothed = data[tcn].rolling(N, 0, center=True, win_type='triang').mean()
            pts = find_all_peaks(smoothed, override=override)
            # if there exists a single peak within the bounds, it is now smooth
            valid_points = pts[(lower <= data[tcn].iloc[pts].values) & \
                               (data[tcn].iloc[pts].values <= upper) & \
                               (left <= time.iloc[pts].values) & \
                               (time.iloc[pts].values <= right)]
            if len(valid_points) <= thresh:
                smooth = True
#                plt.plot(time.iloc[valid_points], smoothed.iloc[valid_points],'k.')
            if N>win_incr*25:
                print('{} reached max smoothing: N = {}'.format(tcn, win_incr*25))
                break
        smooth_data[tcn] = smoothed

    if return_series:
        return smooth_data[tcn]
    else:
        return smooth_data
#%% SMOOTH DEBUG
#import PMG.COM.table as tb
#import PMG.COM.plotstyle as style
#THOR = 'P:/AHEC/Data/THOR/'
#chlist = []
#chlist.extend(['11NECKLO00THFOXA','11NECKLO00THFOYA','11CHSTLEUPTHDSXB','11FEMRLE00THFOZB','11CHSTRILOTHDSXB'])
#chlist.extend(['11HEAD0000THACXA','11SPIN0100THACXC', '11CHST0000THACXC', '11SPIN1200THACXC', '11PELV0000THACXA'])
#chlist.extend(['11HEAD0000THACYA','11SPIN0100THACYC', '11CHST0000THACYC', '11SPIN1200THACYC', '11PELV0000THACYA'])
#chlist.extend(['11SPIN0100THACYC','11THSP0100THAVXA','11THSP0100THAVZA'])
##time, fulldata = import_data(THOR, chlist, check=False)
#table = tb.get('THOR')
#table = table[table.TYPE.isin(['Frontale/Véhicule'])]
#slips  = table[table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
#oks = table[table.CBL_BELT.isin(['OK'])].CIBLE.tolist()
#
#group = oks
#allchannels = [['11HEAD0000THACXA','11SPIN0100THACXC', '11CHST0000THACXC', '11SPIN1200THACXC', '11PELV0000THACXA'],
#               ['11HEAD0000THACYA','11SPIN0100THACYC', '11CHST0000THACYC', '11SPIN1200THACYC', '11PELV0000THACYA']]
##%%
#for channel in allchannels[0]+allchannels[1]:
#
#    plt.close('all')
#    plt.figure()
#    ax = plt.gca()
#    data = fulldata[channel].loc[:,group].dropna(axis=1)
#
##    lower, upper = data.min().min(), data.max().max()
#    positive = abs(data.max().median())>abs(data.min().median())
#    through_zero = data.max().median()*data.min().median()<0
#    if through_zero:
#        lower, upper = (data.max().min()/2, data.max().max()) if positive else (data.min().min(), data.min().max()/2)
#    else:
#        lower, upper = 0
#
#    ax.plot(data, alpha=0.5)
##    ax.axhline(data.max().max())
##    ax.axhline(data.max().median())
##    ax.axhline(data.max().min())
##    ax.axhline(data.min().max())
##    ax.axhline(data.min().median())
##    ax.axhline(data.min().min())
#    ax.axhline(upper, color='b')
#    ax.axhline(lower, color='r')
#    ax.axhline(data.max().min() if positive else data.min().max(), color='g')
#    ax.axhline(data.max().median() if positive else data.min().median(), color='tab:purple')
#
#    plt.waitforbuttonpress()
##%%
#for channel in allchannels[1]:#+allchannels[1]:
#
#    plt.close('all')
#    r, c = style.sqfactors(len(group))
#    fig, axs = style.subplots(r, c, sharex='all', sharey='all')
#    data = fulldata[channel].loc[:,group].dropna(axis=1)
#    smooth_data = pd.DataFrame(columns=data.columns)
#    for ax, tcn in zip(axs, data.columns):
#        N=0
#        smooth=False
#        lower, upper = bounds(data)
#        while not smooth:
#            N+=20
#            smoothed = data[tcn].rolling(N, 0, center=True, win_type='triang').mean()
#            override = 'max' if positive_max(data) else 'min'
#            pts = find_all_peaks(smoothed, override=override)
#            # if there exists a single peak within the bounds, it is now smooth
#            if len(pts[(lower <= data[tcn].loc[pts].values) & \
#                       (data[tcn].loc[pts].values <= upper)]) == 1:
#                smooth = True
#            if N>500:
#                break
#        smooth_data[tcn] = smoothed
#        ax.plot(data[tcn], alpha=0.5)
#        ax.plot(smooth_data[tcn])
#        ax.plot(smooth_data[tcn].loc[pts], 'k.')
#        ax.axhline(upper)
#        ax.axhline(lower)
#
#    plt.waitforbuttonpress()
#%%
def clean_outliers(data, stage, interval=slice(None)):
    """Clean the outliers from the data

    Input
    ---------
    raw : DataFrame, list of DataFrames
        Input raw data
    stage : int 0-2
        Degree of cleaning: 0 for not cleaning, 1 for huge spikes, 2 for far-from-average
    interval : slice
        For degree 2, slice on which the test should be performed. Often the
        first 10 ms and last 200ms can fail the tests but are not in the
        area of interest.

    Returns
    ----------
    clean : DataFrame
        Cleaned data
    """

    if isinstance(data, list):
        data = pd.concat(data, axis=1)
    data = data.T[data.any()].T
    untrimmed = data.copy()

    if stage not in [0,1,2]:
        raise ValueError('Invalid stage selection. Use 0, 1, or 2')

    if stage == 0:
        return data

    if stage >= 1:
        """This stage should recover data that is not the result of defective mesurements"""

#        tcns = []
#        for tcn in data.columns:
#            if (data[tcn]-data[tcn].rolling(3,0,center=True).mean()>10).any():
#                tcns.append(tcn)
#        data = data.drop(tcns, axis=1)
        data = data.loc[:,~(data-data.rolling(3,0,center=True).mean()>10).any()]

        ### Slope Method
#        slopes = np.abs(np.diff(data.T)).max(axis=1)
#        mad = (data.T-data.T.median()).abs().median().T #outlier-adjusted std
#        low, high = data.median(axis=1)-3*mad, data.median(axis=1)+3*mad
#        thresh = (high.max()-low.min())*0.25 #(factor 0.25-0.5)
#        spikes = np.nonzero(slopes>thresh)[0]
#        data = data.drop(data.columns[spikes], axis=1)

    data = data[interval]
    if stage == 2:
        """This stage should take accurate but unruly data and remove out-of-the-ordinary traces"""

        ### ratio of std dev
##        plt.figure()
##        plt.plot(untrimmed, alpha=0.2)
##        tcns = []
##        for tcn in data.columns:
##            if ((data.std(axis=1)/data.drop(tcn,1).std(axis=1))>1.25).sum()>=30:
##                tcns.append(tcn)
##                plt.plot(data[tcn]/(data.std(axis=1)/data.drop(tcn,1).std(axis=1)>1.25))
##        data = data.drop(tcns, axis=1)
##        outliers = data.loc[:,tcns]
##        plt.plot(outliers)
#        std_ratio = data.apply(lambda x: data.std(axis=1)/data.drop(x.name,1).std(axis=1), axis=0)
#        data = data.loc[:,~((std_ratio>1.25).sum()>30).any()]

        ### difference of std dev
        window = {'window':30,'min_periods':0,'center':True,'win_type':'triang'}
        roll = data.rolling(**window).mean()
        std_diff = roll.apply(lambda x: roll.std(axis=1)-roll.drop(x.name,1).std(axis=1), axis=0)
        data = data.loc[:,~(std_diff.rolling(**window).sum()>20).any()]
#        roll = data.rolling(30,0,center=True,win_type='triang').mean()
#        tcns = []
#        for tcn in data.columns:
#            offset = (roll.std(axis=1)-roll.drop(tcn,1).std(axis=1)).rolling(30,0,center=True,win_type='triang').sum()
#            if (offset>=20).any():
#                tcns.append(tcn)
#        data = data.drop(tcns, axis=1)

        ### Outliergram
#        from PMG.outliergram import outliergram
#        data, *_ = outliergram(data, mode='both', factorsh=1.5, factormg=1.5)

    return untrimmed[data.columns]
#%% clean_outliers DEBUG
#tr = 6
#ln = 2
#plt.close('all')
#for ch in chlist:
#    data = fulldata[ch]
#    #data = clean_outliers(data, stage=1)
#    plt.figure()
#    plt.plot(data, alpha=0.2)
#    plt.gca().set_prop_cycle(None)
#    tcns = []
#    for tcn in data.columns:
#        plt.plot(data.std(axis=1)/data.drop(tcn,1).std(axis=1))
#    plt.gca().set_prop_cycle(None)
#    for tcn in data.columns:
#        if ((data.std(axis=1)/data.drop(tcn,1).std(axis=1))>tr).sum()>=ln:
#            tcns.append(tcn)
#            plt.plot(data[tcn]/(data.std(axis=1)/data.drop(tcn,1).std(axis=1)>tr))

#%%

def check_and_clean(raw, stage, interval=slice(None)):
    """Check thats the data does not contain outliers as found by
    clean_outliers(). Prompts the user for whether a plot highlighting the
    outliers should be shown. Options are to retain or discard the outliers.

    See clean_outliers() for details
    """
    clean = clean_outliers(raw,  stage, interval)
    outliers = pd.concat([raw, clean], axis=1).T.drop_duplicates(keep=False).T

    if not outliers.empty:
        action = input("Outliers have been detected.\nview: 'y' \ndiscard: any\nretain: 'keep'\n>>>")
        print(action)
        if action == 'y':
            plt.figure()
            plt.plot(clean, alpha=0.25)
            plt.plot(outliers)
            while not plt.waitforbuttonpress():
                try:
                    pass
                except KeyboardInterrupt:
                    break
            plt.close('all')
            action = input("discard: any\nretain: 'keep'\n>>>")

        if action == 'keep':
            clean = raw
        else:
            print('outlier(s) ignored: '+', '.join(outliers.columns.tolist())+'\n')

    return clean

def tolerance(ax, time, df, color):
    quantiles = np.array([0.2,0.05])#np.arange(0,2+1,1)/2*0.2
    ax.plot(time, df.median(axis=1).rolling(100, 0, center=True, win_type='parzen').mean(),
            color=color, label='Median, n = {}'.format(df.shape[1]))
    for i, alpha in enumerate(quantiles):
        ax.fill_between(time, df.quantile(alpha/2, axis=1).rolling(100, 0, center=True, win_type='parzen').mean(),
                         df.quantile(1-alpha/2, axis=1).rolling(100, 0, center=True, win_type='parzen').mean(),
                         alpha=0.20, color=color, lw=0)
    ax.fill_between(time, float('nan'), float('nan'), color=color, alpha=0.25,
                    label='80th-95th Percentiles')

def stats(df):
    alpha = 0.2 #0.05 = 95% coverage

    df = df.dropna(axis=1)
    N = df.shape[1]

    stats = {'Mid': df.median(axis=1),
             'Low': df.quantile(alpha/2, axis=1),
             'High' : df.quantile(1-alpha/2, axis=1)}
    stats = pd.DataFrame(data=stats)

    not_too_high = (df.T>stats['High']).T.sum().sort_values()[:N-N//10] #remove top 10% of data by time spent outside bounds
    not_too_low = (df.T<stats['Low']).T.sum().sort_values()[:N-N//10]
    between = df[not_too_high.index.intersection(not_too_low.index)]

    stats['Mean-between'] = between.mean(axis=1)

    stats = stats.rolling(100, 0, center=True, win_type='parzen').mean()

    return stats

#def downsample(df, window):
#    #data.set_index(pd.to_datetime(data.index, unit='D')).resample('{}D'.format(5)).last()
#    downsampled = df[::window].copy()
#    for l, r in zip(downsampled.index, downsampled.index+window):
#        downsampled.loc[l,:] = df.loc[l:r,:].mean(axis=0)
#    return downsampled
#
#def upsample(df, window):
#    #data.resample('D').mean().fillna(method='ffill').rolling(5,0,center=True).mean().reset_index()
#    upsampled = pd.DataFrame([], columns=df.columns,
#                             index=range(df.index[0],df.index[-1]+window))
#    for row in df.index:
#        upsampled.loc[row,:] = df.loc[row,:]
#    upsampled = upsampled.fillna(method='ffill')
#    return upsampled.rolling(window, 0, center=True).mean()

if __name__ == '__main__':
    pass
