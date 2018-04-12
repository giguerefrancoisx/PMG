# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:23:18 2018

@author: tangk
"""
import numpy as np
import scipy.signal as signal
import pandas as pd
#from PMG.COM.intersection import intersection
# get properties of curves


# get location of peak:
def get_ipeak(data):
    if np.isnan(data).all():
        return [np.nan, np.nan]
    else:
        return [np.argmin(data), np.argmax(data)] # returns locations of local min and local max

# get value of peak
def peakval(data):
    if np.isnan(data).all():
        return [np.nan, np.nan]
    else:
        return [min(data), max(data)] # returns values of local min and local max

def smooth_data(data):
    if data==[]:
        return []
    elif np.isnan(data).all():
        return np.nan
    return signal.savgol_filter(data,201,5)

def get_i2peak(data):
    if np.isnan(data).all():
        return [np.nan, np.nan]
    if data.size==0:
        return []
    if np.isnan(data).all():
        return [np.nan, np.nan]
    peaks_indices = [signal.find_peaks_cwt(np.negative(data),[50])]
    peaks_indices.append(signal.find_peaks_cwt(data,[50]))
    for i in range(2): # min and max
        if len(peaks_indices[i])>1:
            peaks_vals = data[peaks_indices[i]]
            peaks_indices[i] = peaks_indices[i][abs(peaks_vals)>max(abs(peaks_vals))-5]
            peaks_indices[i] = peaks_indices[i][0]
        elif len(peaks_indices[i])==1:
            peaks_indices[i] = peaks_indices[i][0]
        else:
            peaks_indices[i] = []
    return peaks_indices

def get_t2peak(t,data):
    # input is rearranged i2peak
    out = pd.DataFrame(index=data.index,columns=data.columns)
    for col in data.columns:
        for i in data.index:
            if ~np.isnan(data.get_value(i,col)):
                out.set_value(i,col,t[data.get_value(i,col)])
            else:
                out.set_value(i,col,np.nan)
    return out

def get_Dt2peak(data):
    # data is a props
    channels = list(zip(*list(data['t2peak'])[0::2]))[0]
    files = data['t2peak'].index[:-1]
    cols = data['t2peak'].columns
    rows = pd.MultiIndex.from_product([channels,list(files)+['cdf']])
    out = pd.DataFrame(index=rows,columns=cols)

    for i in range(len(channels)):
        for j in range(i+1,len(channels)):
            for f in files:
                for sign in ['-tive','+tive']:
                    ch1 = channels[i]
                    ch2 = channels[j]
                    t1 = data['t2peak'][ch1][sign][f]
                    t2 = data['t2peak'][ch2][sign][f]
                    out.set_value((ch2,f),(ch1,sign),t1-t2)
    return out

def get_fwhm(t,dataframe):
    out = pd.DataFrame(index=dataframe.index,columns=dataframe.columns)
    for col in dataframe.columns:
        for i in dataframe.index:
            data = dataframe[col][i]
            if np.isnan(data).all():
                out.set_value(i,col,[np.nan,np.nan])
            else:     
                hm1 = np.matlib.repmat(np.min(data)/2,len(data),1)
                hm2 = np.matlib.repmat(np.max(data)/2,len(data),1)
                
                fwhm1 = intersection(t,data,t,hm1)
                fwhm2 = intersection(t,data,t,hm2)
                
                if len(fwhm1[0])<2:
                    fwhm1 = np.nan
                else:
                    fwhm1 = fwhm1[0][-1]-fwhm1[0][0]
                if len(fwhm2[0])<2:
                    fwhm2 = np.nan
                else:
                    fwhm2 = fwhm2[0][-1]-fwhm2[0][0]
                out.set_value(i,col,[fwhm1, fwhm2])
    
    return out

def get_tfwhm(t,dataframe):
    out = pd.DataFrame(index=dataframe.index,columns=dataframe.columns)
    for col in dataframe.columns:
        for i in dataframe.index:
            data = dataframe[col][i]
            
            if np.isnan(data).all():
                out.set_value(i,col,[np.nan,np.nan])
            else:
                hm1 = np.matlib.repmat(np.min(data)/2,len(data),1)
                hm2 = np.matlib.repmat(np.max(data)/2,len(data),1)
                
                fwhm1 = intersection(t,data,t,hm1)
                fwhm2 = intersection(t,data,t,hm2)
                
                if len(fwhm1[0])<2:
                    fwhm1 = np.nan
                else:
                    fwhm1 = [fwhm1[0][0],fwhm1[1][0],fwhm1[0][-1],fwhm1[1][-1]]
                if len(fwhm2[0])<2:
                    fwhm2 = np.nan
                else:
                    fwhm2 = [fwhm2[0][0],fwhm2[1][0],fwhm2[0][-1],fwhm2[1][-1]]
                out.set_value(i,col,[fwhm1, fwhm2])
    return out

def get_mean(data):
    return np.mean(data)

def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4

def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj

def intersection(x1,y1,x2,y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.

usage:
x,y=intersection(x1,y1,x2,y2)

    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)

    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()

    """
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]
