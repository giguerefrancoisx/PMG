from numpy import array, zeros, argmin, inf, ndim, array
from scipy.spatial.distance import cdist

def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def fastdtw(x, y, w, dist='euclidean'):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the warp path.
    """
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    mask = zeros([r, c])
    for i in range(r):
        for j in range(max(0, i-w), min(c, i+w+1)):
            mask[i, j] = 1
    mask[mask==0] = inf #change 0 to inf without dividing by zero
    D0[1:,1:] = cdist(x,y,dist)*mask
    C = D1.copy()
    for i in range(r):
        for j in range(max(0, i-w), min(c, i+w+1)):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

if __name__ == '__main__':
    if 0: # 1-D numeric
        from sklearn.metrics.pairwise import manhattan_distances
        x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
        y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
        dist_fun = manhattan_distances
    elif 1: # 2-D numeric
        from sklearn.metrics.pairwise import euclidean_distances
        x = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]]
        y = [[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]]
        x = array(x)
        y = array(y)
        dist_fun = euclidean_distances
    else: # 1-D list of strings
        from nltk.metrics.distance import edit_distance
        #x = ['we', 'shelled', 'clams', 'for', 'the', 'chowder']
        #y = ['class', 'too']
        x = ['i', 'soon', 'found', 'myself', 'muttering', 'to', 'the', 'walls']
        y = ['see', 'drown', 'himself']
        #x = 'we talked about the situation'.split()
        #y = 'we talked about the situation'.split()
        dist_fun = edit_distance
    w=len(x)
    dist, cost, acc, path = fastdtw(x, y, w)

    # vizualize
    from matplotlib import pyplot as plt
    plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
    plt.plot(path[0], path[1], '-o') # relation
    plt.xticks(range(len(x)), x)
    plt.yticks(range(len(y)), y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('tight')
    plt.title('Minimum distance: {}'.format(dist))
    plt.show()
#%% 8 12 13 17
#import numpy as np
#from matplotlib import pyplot as plt
#plt.close('all')
#x = X[13]
#y = X[17]
#t = np.array(time[sl][::5])
##distance, D = twed(x,t,y,t,1,1)
##path = backtracking(D)
##D_mat = {}
##for w in [146,100,40,0]:
#if 1==1:
#    w=146
#    distance, C, D, path = fastdtw(x,y,w)
##    D_mat[w] = D
#    print(distance)
##
#    plt.figure()
#    plt.imshow(D.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
#    plt.plot(path[0], path[1], '-') # relation
#    plt.xticks(range(len(x)), x)
#    plt.yticks(range(len(y)), y)
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.axis('tight')
#    plt.title('Minimum distance: {}'.format(distance))
#    plt.show()
#
#    xi, yi = path
#    x1 = x[xi]
#    y1 = y[yi]
#    plt.figure()
#    plt.plot(t, x)
#    plt.plot(t, y)
#    plt.plot(time[sl.start:sl.start+5*len(x1)][::5], x1, label='x1')
#    plt.plot(time[sl.start:sl.start+5*len(y1)][::5], y1, label='y1')
#    plt.legend()
