# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:23:18 2018

@author: tangk
"""
import numpy as np
import scipy.signal as signal
import pandas as pd
from PMG.COM.intersection import intersection
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
        return np.nan
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