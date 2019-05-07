# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:23:18 2018

@author: tangk
"""
import numpy as np
import scipy.signal as signal
import pandas as pd
from scipy.integrate import trapz, simps
#from PMG.COM.intersection import intersection
# get properties of curves

def get_min(data):
    if np.isnan(data).all():
        return np.nan
    else:
        return min(data)
    
def get_max(data):
    if np.isnan(data).all():
        return np.nan
    else:
        return max(data)

def get_argmin(data):
    if np.isnan(data).all():
        return np.nan
    else:
        return np.argmin(data)
    
def get_argmax(data):
    if np.isnan(data).all():
        return np.nan
    else:
        return np.argmax(data)

def smooth_data(data):
    if np.isnan(data).all():
        return data
    else:
        return signal.savgol_filter(data,201,5)

def get_angle(row):
    # angle from the vertical 
    dx = np.abs(row['Up_x']-row['Down_x'])
    dy = np.abs(row['Up_y']-row['Down_y'])
    angle = np.degrees(np.arctan(dx/dy))
    return angle

def get_i_to_t(t):
    def i_to_t(i):
        if np.isnan(i):
            return np.nan
        else:
            return t[int(i)]
    return i_to_t


def get_auc(data, dt=1/10000):
    if np.isnan(data).all():
        return np.nan
    else:
        return trapz(data, dx=dt)


def get_x_at_y(x, y):
    """x is a pd.Series or pd.DataFrame of time series (like chdata)
    y is a pd.Series of indices. Returns the value of each element of x 
    at the corresponding index specified in y."""
    if isinstance(x, pd.core.series.Series):
        x = pd.DataFrame(x)
    y = y.dropna().astype(int)
    
    out = pd.DataFrame(np.nan, columns=x.columns, index=x.index)
    for i in y.index:
        out.loc[i] = x.loc[i].apply(lambda x: x[y[i]])
    if out.shape[1]==1:
        out = out.iloc[:,0]
    return out


def get_onset_to_max(data, pct=0.1, thresh=None, filt=None):
    """data is an array-like. gets the first i at which the value of data 
    exceeds max(data)*pct or the threshold specified by thresh. If both pct 
    and thresh are given, pct is used to determine ionset.
    
    If filter is given, the logical is put through the filter before
    finding the first index where the value of data exceeds the threshold."""
    if np.isnan(data).all():
        return np.nan
    
    if pct:
        thresh = pct*max(data)
    over_thresh = data > thresh
    if filt:
        over_thresh = filt(over_thresh)
    return np.argmax(over_thresh)

   
def get_onset_to_min(data, pct=0.1, thresh=None, filt=None):
    """does the same as get_onset_to_max but assumes the peak is negative"""
    return get_onset_to_max(-data, pct=pct, thresh=thresh, filt=filt)
    

def get_peaks(chdata):
    """
    features = get_peaks(chdata)
    Returns the max and min values of all time series in chdata. 
    """
    feature_funs = {'Min_': [get_min],
                    'Max_': [get_max]} 
    features = pd.concat(chdata.chdata.get_features(feature_funs).values(),axis=1,sort=True)
    return features


#def get_i2peak(data):
#    if np.isnan(data).all():
#        return [np.nan, np.nan]
#    if data.size==0:
#        return []
#    if np.isnan(data).all():
#        return [np.nan, np.nan]
#    peaks_indices = [signal.find_peaks_cwt(np.negative(data),[50])]
#    peaks_indices.append(signal.find_peaks_cwt(data,[50]))
#    for i in range(2): # min and max
#        if len(peaks_indices[i])>1:
#            peaks_vals = data[peaks_indices[i]]
#            peaks_indices[i] = peaks_indices[i][abs(peaks_vals)>max(abs(peaks_vals))-5]
#            peaks_indices[i] = peaks_indices[i][0]
#        elif len(peaks_indices[i])==1:
#            peaks_indices[i] = peaks_indices[i][0]
#        else:
#            peaks_indices[i] = np.nan
#    return peaks_indices
#
#def get_t2peak(t,data):
#    # input is rearranged i2peak
#    return data.applymap(lambda x: t[int(x)] if not np.isnan(x) else np.nan)
#    
#    out = pd.DataFrame(index=data.index,columns=data.columns)
#    for col in data.columns:
#        for i in data.index:
#            if ~np.isnan(data.at[i,col]):
#                out.at[i,col] = t[data.at[i,col]]
#            else:
#                out.at[i,col] = np.nan
#    return out

#def get_Dt2peak(data):
#    # data is a props
#    channels = list(zip(*list(data['t2peak'])[0::2]))[0]
#    files = data['t2peak'].index[:-1]
#    cols = data['t2peak'].columns
#    rows = pd.MultiIndex.from_product([channels,list(files)+['cdf']])
#    out = pd.DataFrame(index=rows,columns=cols)
#
#    for i in range(len(channels)):
#        for j in range(i+1,len(channels)):
#            for f in files:
#                for sign in ['-tive','+tive']:
#                    ch1 = channels[i]
#                    ch2 = channels[j]
#                    t1 = data['t2peak'][ch1][sign][f]
#                    t2 = data['t2peak'][ch2][sign][f]
#                    out.set_value((ch2,f),(ch1,sign),t1-t2)
#    return out
#
#def get_Dpeak(data):
#    # data is a props
#    channels = list(zip(*list(data['peak'])[0::2]))[0]
#    files = data['peak'].index[:-1]
#    cols = data['peak'].columns
#    rows = pd.MultiIndex.from_product([channels,list(files)+['cdf']])
#    out = pd.DataFrame(index=rows,columns=cols)
#
#    for i in range(len(channels)):
#        for j in range(i+1,len(channels)):
#            for f in files:
#                for sign in ['-tive','+tive']:
#                    ch1 = channels[i]
#                    ch2 = channels[j]
#                    t1 = data['peak'][ch1][sign][f]
#                    t2 = data['peak'][ch2][sign][f]
#                    out.set_value((ch2,f),(ch1,sign),t1-t2)
#    return out


#def get_fwhm(t,dataframe):
#    out = pd.DataFrame(index=dataframe.index,columns=dataframe.columns)
#    for col in dataframe.columns:
#        for i in dataframe.index:
#            data = dataframe[col][i]
#            if np.isnan(data).all():
#                out.set_value(i,col,[np.nan,np.nan])
#            else:     
#                hm1 = np.matlib.repmat(np.min(data)/2,len(data),1)
#                hm2 = np.matlib.repmat(np.max(data)/2,len(data),1)
#                
#                fwhm1 = intersection(t,data,t,hm1)
#                fwhm2 = intersection(t,data,t,hm2)
#                
#                if len(fwhm1[0])<2:
#                    fwhm1 = np.nan
#                else:
#                    fwhm1 = fwhm1[0][-1]-fwhm1[0][0]
#                if len(fwhm2[0])<2:
#                    fwhm2 = np.nan
#                else:
#                    fwhm2 = fwhm2[0][-1]-fwhm2[0][0]
#                out.set_value(i,col,[fwhm1, fwhm2])
#    
#    return out

#def get_tfwhm(t,dataframe):
#    out = pd.DataFrame(index=dataframe.index,columns=dataframe.columns)
#    for col in dataframe.columns:
#        for i in dataframe.index:
#            data = dataframe[col][i]
#            
#            if np.isnan(data).all():
#                out.set_value(i,col,[np.nan,np.nan])
#            else:
#                hm1 = np.matlib.repmat(np.min(data)/2,len(data),1)
#                hm2 = np.matlib.repmat(np.max(data)/2,len(data),1)
#                
#                fwhm1 = intersection(t,data,t,hm1)
#                fwhm2 = intersection(t,data,t,hm2)
#                
#                if len(fwhm1[0])<2:
#                    fwhm1 = np.nan
#                else:
#                    fwhm1 = [fwhm1[0][0],fwhm1[1][0],fwhm1[0][-1],fwhm1[1][-1]]
#                if len(fwhm2[0])<2:
#                    fwhm2 = np.nan
#                else:
#                    fwhm2 = [fwhm2[0][0],fwhm2[1][0],fwhm2[0][-1],fwhm2[1][-1]]
#                out.set_value(i,col,[fwhm1, fwhm2])
#    return out

#def get_tonset(t,dataframe):
#    # dataframe is in the form of chdata
#    out = pd.DataFrame(index=dataframe.index,columns=dataframe.columns)
#    for col in dataframe.columns:
#        for i in dataframe.index:
#            data = dataframe.at[i,col]
#            if np.isnan(data).all():
#                out.at[i,col] = [np.nan,np.nan]
#            else:
#                onset_pos = np.repeat(0.1*max(data),len(t))
#                onset_neg = np.repeat(0.1*min(data),len(t))
#                
#                tonset_pos = intersection(t,data,t,onset_pos)
#                tonset_neg = intersection(t,data,t,onset_neg)
#                
#                out.at[i,col] = (tonset_neg[0][0], tonset_pos[0][0])
#    return out

#def get_shifted(row):
#    # row is in the form of a pandas series
#    # shifts 
#    row[['Down_x','Up_x']] = row[['Down_x','Up_x']] - row['Down_x'][0]
#    row[['Down_y','Up_y']] = row[['Down_y','Up_y']] - row['Down_y'][0]
#    return row

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


