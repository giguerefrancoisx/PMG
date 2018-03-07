# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:08:58 2018

@author: giguerf
"""
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt

def outliergram(df, mode='magnitude', report=False, factorsh=1.5, factormg=2.5):

    data = np.array(df)
    x = data.T
    n, p = x.shape

    rmat = np.apply_along_axis(lambda a: scipy.stats.rankdata(a, method='max'), arr=x, axis=0)
    down = rmat - 1
    up = n - rmat

    mbd = (np.sum(up*down, axis=1)/p+n-1)/scipy.special.binom(n, 2)
    epi = np.sum(up+1, axis=1)/(n*p)

    if mode in ['shape','both']:
        a0 = -2/(n*(n-1))
        a1 = 2*(n+1)/(n-1)
        a2 = a0

        P = np.poly1d([a2*n**2, a1, a0])
        d = P(epi)-mbd
        Q = np.percentile(d, 75)+factorsh*np.diff(np.percentile(d, [25,75]))

        shape_outliers = np.nonzero(d>Q)[0]
        shape = df.iloc[:,shape_outliers]

    if mode in ['magnitude','both']:
        m = np.ceil(n/2).astype(int)
        center = x[np.argsort(mbd)[m:n],:]
        inf = np.min(center, axis=0)
        sup = np.max(center, axis=0)
        dist = factormg*(sup-inf)
        upper = sup+dist
        lower = inf-dist

        upper = np.array([upper]*n)
        lower = np.array([lower]*n)

        outside = (x<lower)+(x>upper)
        outrow = np.sum(outside, axis=1)
        magnitude_outliers = np.nonzero(outrow>100)[0] #50 = threshold for consideration (time outside bounds)
        magnitude = df.iloc[:,magnitude_outliers]

    if report:
        print('SHAPE OUTLIERS')
        print('outliers idx:', shape_outliers)
        print('outliers epi:', epi[shape_outliers])
        print('outliers mbd:', mbd[shape_outliers])

        print('MAGNITUDE OUTLIERS')
        print('outside count:', outrow)
        print('outliers idx:', magnitude_outliers)
        print('outliers epi:', epi[magnitude_outliers])
        print('outliers mbd:', mbd[magnitude_outliers])

        poly = np.polyfit(epi, mbd, 2)
        fit = np.poly1d(poly)
        values = np.linspace(0, 1, 100)

        plt.close('all')
        plt.figure()
        plt.plot(epi, mbd, '.')
        plt.plot(epi[shape_outliers], mbd[shape_outliers], '.', label='shp')
        plt.plot(epi[magnitude_outliers], mbd[magnitude_outliers], '.', label='mag')
        plt.plot(values, fit(values), label='fit')
        plt.plot(values, P(values), label='P')
        plt.plot(values, (P-Q)(values), label='P-Q')
        plt.legend()
        plt.ylim(0,1.08*max(mbd))

        plt.figure()
        plt.plot(data, alpha=0.1)
        plt.plot(lower[0], label='lower')
        plt.plot(upper[0], label='upper')
        if shape_outliers.size != 0:
            plt.plot(data[:,shape_outliers])
        if magnitude_outliers.size != 0:
            plt.plot(data[:,magnitude_outliers])
        plt.legend()

    if mode=='shape':
        safe = pd.concat([df, shape], axis=1).T.drop_duplicates(keep=False).T
        return safe, shape
    elif mode == 'magnitude':
        safe = pd.concat([df, magnitude], axis=1).T.drop_duplicates(keep=False).T
        return safe, magnitude
    elif mode == 'both':
        safe = pd.concat([df, shape, magnitude], axis=1).T.drop_duplicates(keep=False).T
        return safe, shape, magnitude
    else:
        raise ValueError('Invalid mode selection')

if __name__ == '__main__':
    new_df = df.loc[:,:].dropna(axis=1).rolling(window=100, center=True, min_periods=0, win_type='parzen').mean()
    safe, *outliers = outliergram(new_df, mode='both', report=True, factorsh=1.5, factormg=2.5)
