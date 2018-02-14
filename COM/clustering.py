# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:45:52 2018

@author: giguerf
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import cdist, pdist
from PMG.COM.data import import_data, process
from PMG.COM.plotstyle import sqfactors, subplots

def make_labels(raw, project, tcns):

    if project =='AHEC':
        table = pd.read_excel('P:/AHEC/ahectable.xlsx')
        table2 = table.copy()
        table2.columns = table.columns[3:6].tolist()+table.columns[0:3].tolist()+table.columns[6:].tolist()
        table = pd.concat([table, table2])
        column = 'CIBLE'
        keys = ['SUBSET', 'VITESSE', 'TCN', 'CBL_MODELE']
        shortkeys = ['TCN', 'CBL_MODELE']
        slicer = dict(zip(keys, [slice(3),slice(None),slice(None),slice(None)]))

    elif project == 'THOR':
        table = pd.read_excel('P:/AHEC/thortable.xlsx', sheetname='All')
        column = 'CIBLE'
        keys = ['CBL_BELT', 'TCN', 'CBL_MODELE']
        shortkeys = ['CBL_BELT', 'TCN', 'CBL_MODELE']
        slicer = dict(zip(keys, [slice(2),slice(None),slice(None)]))

    elif project == 'BOOSTER':
        table = pd.read_excel('P:/BOOSTER/boostertable.xlsx')
        column = 'TC_Vehicule'
        keys = ['Group','TCN','Marque', 'Modele']
        shortkeys = ['TCN','Group','Marque', 'Modele']
        slicer = dict(zip(keys, [slice(None),slice(None),slice(None),slice(1)]))

    else:
        raise(ValueError('project not available for selection'))

    labels = raw.columns.tolist()
    lbd = {}
    for i, tcn in enumerate(labels):
        attr = {}
        attr['TCN'] = tcn
        try:
            for key in list(set(keys)-set(['TCN'])):
                series = table[table.loc[:,[column]].isin([tcn]).any(axis=1)][key]
                attr[key] = str(series.iloc[0])[slicer[key]]
            labels[i] = ' '.join([attr[key] for key in keys]).upper()
            lbd[tcn] =  ' '.join([attr[key] for key in shortkeys]).upper()
        except IndexError:
            labels[i] = 'Error'
            lbd[tcn] = 'Error'

    return labels, lbd

def fastdtw(x, y, w=None, dist='euclidean'):
    tau = np.arange(len(x))/len(x)
    tx, x = respace(tau, x)
    ty, y = respace(tau, y)
    x = np.vstack([tx, x]).transpose()
    y = np.vstack([ty, y]).transpose()
    if w is None:
        w = len(x)
    D0 = np.zeros((len(x) + 1, len(y) + 1))
    D0[0, 1:] = float('inf')
    D0[1:, 0] = float('inf')
    D = D0[1:, 1:]
    mask = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(max(0, i-w), min(len(y), i+w+1)):
            mask[i, j] = 1
    mask[mask==0] = float('inf') #change 0 to inf without dividing by zero
    D0[1:,1:] = cdist(x,y,dist)*mask
    for i in range(len(x)):
        for j in range(max(0, i-w), min(len(y), i+w)):
            D[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
#    plot_preview(x, y, D0)
    return D[-1, -1]/sum(D.shape)

def dtw_dist(x, y):
    global count
    print(next(count)+1)
    return fastdtw(x, y, w=len(x)//2+1)

def dist_matrix(Y, labels):
    m = int((2*len(Y)+1/4)**0.5+1/2)
    T = np.zeros((m,m))
    k = 0
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            T[j,i] = Y[(m-1)*i+j-i*(i+1)//2-1]
            k = k+1
    with open(os.path.expanduser('~/Documents/T.dst'), 'w') as f:
        f.write('{}\tlabelled\n'.format(m))
        for i, long_row in enumerate(T):
            unique, indices = np.unique(long_row, return_index=True)
            arr = np.vstack((indices, unique)).transpose()
            row = arr[arr[:,0].argsort()][:,1]
            line = labels[i]+'\t'+'\t'.join('0' if value == 0 else str(value) for value in row)
            f.write(line + '\n')

def plot_clusters(raw, t, Z, N, name, lbd, labels, plot_all, output):

    results = hierarchy.fcluster(Z, t=N, criterion='maxclust')
    s = pd.DataFrame(results)
    s.index = raw.columns
    s.columns = ['name']
    tcns = {}
    for cluster in s['name'].unique():
        tcns[cluster] = s[s['name']==cluster].index.tolist()

    if plot_all:
        plot_dendrograms(Z, name, labels, output)
        r, c = sqfactors(N)
        fig, axs = subplots(r, c, sharey='all', sharex='all',
                            num='_'.join([name,'cluster']), figsize=[6*c,6*r])
        for i, cluster in enumerate(tcns.keys()):
            ax = axs[i]
            for tcn in tcns[cluster]:
                ax.plot(t, raw.loc[:,tcn], label=lbd[tcn])
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output+name+'_cluster.png')
        plt.close('all')

    return tcns

def plot_dendrograms(Z, name, labels, output):
    plt.figure(name, figsize=[16, 12])
    hierarchy.dendrogram(Z, orientation='left', leaf_font_size=10,
                         labels=labels, distance_sort='ascending')
    plt.tight_layout()
    plt.savefig(output+name+'_dendrogram.png')

def plot_preview(x, y, D0):
    from PMG.COM.dtw import _traceback
    global count
    n = str(count.__length_hint__()+1)
    path = _traceback(D0)
    xi, yi = path
    span = np.arange(len(xi))/len(x)
    plt.close('all')
    x = x.transpose()
    y = y.transpose()
    plt.plot(*x)
    plt.plot(*y)
    plt.plot(span, x[1][xi])
    plt.plot(span, y[1][yi])
    plt.savefig('P:/AHEC/Plots/Clustering/temp/'+n+'p.png')
    plt.close('all')
    plt.plot(*x)
    plt.plot(*y)
    for xii, yii in zip(xi,yi):
        plt.plot([x[0][xii], y[0][yii]], [x[1][xii], y[1][yii]])
    plt.savefig('P:/AHEC/Plots/Clustering/temp/'+n+'d.png')
    plt.close('all')

def respace(t, x, N=None):
    if N is None:
        N = len(t[::5])
    arclength = np.zeros(len(t))
    for i in range(1,len(t)):
        ds = ((t[i]-t[i-1])**2+(x[i]-x[i-1])**2)**0.5
        arclength[i] = arclength[i-1]+ds
    step = arclength[-1]/N
    total = step
    new_points = [0]
    for i, s in enumerate(arclength):
        if s>=total:
            new_points.append(i)
            total += step
    new_points = np.array(new_points)
    t = t[new_points]
    x = x[new_points]
    return t, x

def preprocess(raw, norm=True, smooth=True):
    datadf = raw.copy()
    if norm:
        datadf = (datadf - datadf.mean())/datadf.std()
    if smooth:
        datadf = datadf.rolling(window=30, center=True, min_periods=0).mean()
    sign = int(abs(datadf.max().max())<abs(datadf.min().min()))
    datadf = (datadf - datadf.min().min())/(datadf.max().max() - datadf.min().min())-sign
    return datadf

def cluster(channel, sl, data, project, output, N=6, tcns=None, norm=True,
            smooth=True, plot_all=True, plot_data=True, matrix=False, tag='dtw'):
    """Clusters the data provided into N groups.

    Inputs:
    ----------
    channel:
        ISO code for channel of interest. ex: '11PELV0000THACXA'
    sl:
        slice object for the time interval of interest. ex: slice(200,1650)
    data:
        location of the data folder from which to open HDF5 store. ex: 'P:/AHEC/DATA/'
    project:
        one of 'AHEC', 'BOOSTER', 'THOR'
    output:
        directory in which to save the clustering plots. ex: 'P:/AHEC/Plots/Clustering/'
    N:
        number of clusters to form, integer

    tcns:
        list of TCNs to narrow down the data
    norm:
        bool, whether or not to perform z-normalization
    smooth:
        bool, whether or not to perform smoothing
    plot_all:
        whether or not to make and save plots
    plot_data:
        whether or not to plot data overview after pre-processing
    matrix:
        whether or not to save distance matrix to file ('Documents/T.dst')
    tag:
        label for plot in output (useful for comparing settings)

    Returns
    ----------
    clusters:
        dictionary of clusters by linkage method
    raw:
        raw data, minus outliers and dead channels
    data:
        processed data

    """

#    t, raw, labels, lbd = import_data(channel, sl, SAI, project, tcns)
#    datadf = preprocess(raw, norm, smooth)
#    from GitHub.COM.data import import_SAI, process
    t, raw = import_data(data, channel, tcns, sl)
    labels, lbd = make_labels(raw, project, tcns)
    datadf = process(raw, norm, smooth)
    X = np.array(datadf.transpose())

    m,_ = X.shape
    global count
    count = reversed(range(int(m*(m-1)/2)))

    Yd = pdist(X, metric=dtw_dist)
#    Ye = pdist(X, metric='euclidean')

    if matrix:
        dist_matrix(Yd, labels)

    plt.close('all')

    clusters = {}
    #for metric, Y in zip(['d','e'],[Yd, Ye]):
    for metric, Y in zip([tag],[Yd]):
        for method in ['centroid', 'ward', 'average', 'weighted', 'complete']:
            Z = hierarchy.linkage(Y, method=method)
            name = '_'.join([method, metric])
            clusters[method] = plot_clusters(raw, t, Z, N, name, lbd, labels, plot_all, output)
#            clusters[method] = plot_clusters(datadf, t, Z, N, name, lbd, labels, plot_all, output)

    if plot_data:
        plt.figure()
        ax = plt.gca()
        ax.plot(t, raw)
        ax2 = plt.twinx(ax=ax)
        ax2.plot(t, datadf, ':')

    return clusters, raw, datadf

### Run the main function unless imported
if __name__ == '__main__':
#    chlist = ['10CVEHCG0000ACXD', '10SIMELE00INACXD', '10SIMERI00INACXD',
#              '11CHST0000H3ACXC', '11CHST0000THACXC', '11NECKLO00THFOXA',
#              '11PELV0000H3ACXA', '11PELV0000THACXA']
#    channel = chlist[7]
    channel = '11FEMRLE00THFOZB'
    sl = slice(100,1600)
    data = os.fspath('P:/AHEC/DATA/THOR/')
    project = 'THOR'
    output = 'P:/AHEC/Plots/Clustering/THOR/'
#    tcns = ['TC15-163', 'TC11-008', 'TC09-027', 'TC14-035', 'TC13-007',
#            'TC12-003', 'TC17-201', 'TC17-209', 'TC17-212', 'TC15-162',
#            'TC12-217', 'TC14-220', 'TC12-501', 'TC14-139', 'TC16-013',
#            'TC14-180', 'TC17-211', 'TC16-129', 'TC17-025', 'TC17-208']
    table = pd.read_excel('P:/AHEC/thortable.xlsx', sheetname='All')
    tcns = table[table['CBL_BELT'].isin(['SLIP','OK']) & table['TYPE'].isin(['Frontale/Véhicule'])]['CIBLE'].tolist()

#    tcns=None

    clusters, *data = cluster(channel, sl, data, project, output, N=4, tcns=tcns,
                       norm=False, smooth=True, plot_all=True, plot_data=True,
                       matrix=False, tag='FEMRL-n')

#%% Time stuff
#import time as timer
#t0 = timer.time()
#Y0 = pdist(X, metric=lambda x,y: fastdtw(x,y,w=291))
#t1 = timer.time()
#Y1 = pdist(X, metric=lambda x,y: fastdtw(x,y,w=146))
#t2 = timer.time()
#Y2 = pdist(X, metric=lambda x,y: fastdtw(x,y,w=40))
#t3 = timer.time()
#print('Window1: {}\nWindow2: {}\nWindow3: {}'.format(t1-t0,t2-t1,t3-t2))

#%% Plot two tests for manual comaprison
#from GitHub.COM.dtw import fastdtw as fastdtw_path
#t2 = time
#sl = slice(200,1651)
#tcns = ['TC17-025', 'TC17-028']
#df = (raw.loc[:,tcns] - raw.loc[:,tcns].min().min())/(raw.loc[:,tcns].max().max() - raw.loc[:,tcns].min().min())-1
#df = df.rolling(window=30, center=True, min_periods=0).mean()
#x, y = np.array(df.transpose())
##w=len(x)//2+1
#w=len(x)
##w=40
#tau = np.arange(len(x))/len(x)
#xt = np.vstack([tau, x]).transpose()
#yt = np.vstack([tau, y]).transpose()
#distance, C, D1, path = fastdtw_path(xt, yt, w, 'euclidean')
#xi, yi = path
#x1 = x[xi]
#y1 = y[yi]
#plt.close('all')
#plt.figure()
#plt.plot(t2[sl], x, label=tcns[0])
#plt.plot(t2[sl], y, label=tcns[1])
#plt.plot(t2[sl.start:len(x1)+sl.start], x1, label='x1')
#plt.plot(t2[sl.start:len(x1)+sl.start], y1, label='y1')
#plt.legend()
#
#plt.figure()
#for tcn in tcns:
#    plt.plot(t, datadf.loc[:,tcn], label=tcn)
#plt.legend()
#
#span = np.arange(len(xi))/len(xt)
#xt = xt.transpose()
#yt = yt.transpose()
#plt.figure()
#plt.plot(*xt)
#plt.plot(*yt)
#for xii, yii in zip(xi,yi):
#    plt.plot([xt[0][xii], yt[0][yii]], [xt[1][xii], yt[1][yii]])
