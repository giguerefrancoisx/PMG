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

def make_labels(raw, project):

    if project =='AHEC':
        table = pd.read_excel('P:/AHEC/ahectable.xlsx')
        table2 = table.copy()
        table2.columns = table.columns[3:6].tolist()+table.columns[0:3].tolist()+table.columns[6:].tolist()
        table = pd.concat([table, table2])
        column = 'CIBLE'
        keys = ['TCN', 'CBL_MODELE']
#        keys = ['SUBSET', 'VITESSE', 'TCN', 'CBL_MODELE']
        slicer = dict(zip(keys, [slice(3),slice(None),slice(None),slice(None)]))

    elif project == 'THOR':
        table = pd.read_excel('P:/AHEC/thortable.xlsx', sheetname='All')
        column = 'CIBLE'
        keys = ['CBL_BELT', 'TCN', 'CBL_MODELE']
        slicer = dict(zip(keys, [slice(2),slice(None),slice(None)]))

    elif project == 'BOOSTER':
        table = pd.read_excel('P:/BOOSTER/boostertable.xlsx')
        column = 'TC_Vehicule'
        keys = ['Group','TCN','Marque', 'Modele']
        slicer = dict(zip(keys, [slice(None),slice(None),slice(None),slice(1)]))

    else:
        raise(ValueError('project not available for selection'))

    labels = raw.columns.tolist()
    for i, tcn in enumerate(labels):
        attr = {}
        attr['TCN'] = tcn
        try:
            for key in list(set(keys)-set(['TCN'])):
                series = table[table.loc[:,[column]].isin([tcn]).any(axis=1)][key]
                attr[key] = str(series.iloc[0])[slicer[key]]
            labels[i] = ' '.join([attr[key] for key in keys]).upper()
        except IndexError:
            labels[i] = 'Error'

    return labels

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

def plot_clusters(raw, t, Z, N, name, labels, plot_all, plot_path):

    results = hierarchy.fcluster(Z, t=N, criterion='maxclust')
    s = pd.DataFrame(results)
    s.index = raw.columns
    s.columns = ['name']
    lbd = dict(zip(s.index, labels))
    tcns = {}
    for cluster in s['name'].unique():
        tcns[cluster] = s[s['name']==cluster].index.tolist()

    if plot_all:
        plot_dendrograms(Z, name, labels, plot_path)
        r, c = sqfactors(N)
        fig, axs = subplots(r, c, sharey='all', sharex='all',
                            num='_'.join([name,'cluster']), figsize=[6*c,6*r])
        for i, cluster in enumerate(tcns.keys()):
            ax = axs[i]
            for tcn in tcns[cluster]:
                ax.plot(t, raw.loc[:,tcn], label=lbd[tcn])
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_path+name+'_cluster.png')
        plt.close('all')

    return tcns

def plot_dendrograms(Z, name, labels, plot_path):
    plt.figure(name, figsize=[5, 3])
    hierarchy.dendrogram(Z, orientation='left', leaf_font_size=6,
                         labels=labels, distance_sort='ascending')
    plt.tight_layout()
    plt.savefig(plot_path+name+'_dendrogram.png')

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

def cluster(t, raw, labels, plot_path, mode='dtw', N=4, norm=True, smooth=True,
            plot_all=True, plot_data=True, matrix=False, tag=None):
    """Clusters the data provided into N groups.

    Inputs:
    ----------
    t : Series
        time channel
    raw : DataFrame
        raw, unprocessed data
    plot_path : str
        directory in which to save the clustering plots. ex: 'P:/AHEC/Plots/Clustering/'
    mode : 'dtw', 'euclidean', or 'both'
        Metric(s) to use
    N : int
        number of clusters to form
    norm :
        bool, whether or not to perform z-normalization
    smooth :
        bool, whether or not to perform smoothing
    plot_all :
        whether or not to make and save plots
    plot_data :
        whether or not to plot data overview after pre-processing
    matrix :
        whether or not to save distance matrix to file ('Documents/T.dst')
    tag :
        label for plot in output (useful for comparing settings)

    Returns
    ----------
    clusters :
        dictionary of clusters by linkage method
    data :
        processed data
    """

    df = process(raw, norm, smooth, scale=True)
    X = np.array(df.T)

    m,_ = X.shape
    global count
    count = reversed(range(int(m*(m-1)/2)))

    tag = '_'+tag if tag is not None else ''
    if mode in ['dtw']:
        Yd = pdist(X, metric=dtw_dist)
        distances = zip(['d'+tag],[Yd])
    if mode in ['euclidean']:
        Ye = pdist(X, metric='euclidean')
        distances = zip(['e'+tag],[Ye])
    if mode in ['both']:
        Yd = pdist(X, metric=dtw_dist)
        Ye = pdist(X, metric='euclidean')
        distances = zip(['d'+tag,'e'+tag],[Yd, Ye])

    if matrix:
        dist_matrix(Yd, labels)

    plt.close('all')

    clusters = {}

    for metric, Y in distances:
        for method in ['centroid', 'ward', 'average', 'weighted', 'complete']:
            Z = hierarchy.linkage(Y, method=method)
            name = '_'.join([method, metric])
            clusters[method] = plot_clusters(raw, t, Z, N, name, labels, plot_all, plot_path)
            #Plot processed data instead in clusters:
            #clusters[method] = plot_clusters(datadf, t, Z, N, name, lbd, labels, plot_all, plot_path)

    if plot_data:
        plt.figure()
        ax = plt.gca()
        ax.plot(t, raw)
        ax2 = plt.twinx(ax=ax)
        ax2.plot(t, df, ':')

    return clusters, df

### Run the main function unless imported
if __name__ == '__main__':
    from PMG.COM import table as tb

    path = os.fspath('P:/AHEC/DATA/Full Sample/48/')
    project = 'AHEC'
    plot_path = 'P:/AHEC/Plots/Clustering/THOR/'

    table = tb.get(project)
    tcns = table[table.SUBSET.isin(['HEV vs ICE']) & table.VITESSE.isin([48])].CIBLE.tolist()+table[table.SUBSET.isin(['HEV vs ICE']) & table.VITESSE.isin([48])].BELIER.tolist()
#    tcns = None

    time, raw = import_data(path, '10CVEHCG0000ACXD', tcns, sl=slice(100,1600))
    labels = make_labels(raw, project)

    clusters, data = cluster(time, raw, labels, plot_path, mode='both', N=4,
                             norm=True, smooth=True, plot_all=True,
                             plot_data=True, matrix=False, tag='new')

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
