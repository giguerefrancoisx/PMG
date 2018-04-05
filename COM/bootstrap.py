# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:02:50 2018

@author: tangk
"""
import numpy as np

def bootstrap_resample(X, n=1):
    out = np.zeros((n,len(X)))
    for i in range(n):    
        resample_i = np.floor(np.random.rand(len(X))*len(X)).astype(int)
        X_resample = X[resample_i]
        out[i] = X_resample
    return out

def get_ci(X,X_resample,alpha):
    theta_hat = np.mean(X) # mean of original sample
    SE = np.std(X)/np.sqrt(len(X)) # standard error of original sample
    SE_b = np.std(X_resample,axis=1)/np.sqrt(len(X)) # bootstrap standard error
    t = (np.mean(X_resample,axis=1) - theta_hat)/SE_b # t value
    L = theta_hat - SE*np.percentile(t,(1-alpha/2)*100) # lower bound confidence interval
    U = theta_hat - SE*np.percentile(t,alpha/2*100) # upper bound confidene interval
    return L, U

def get_bins_from_ci(data,ci,nbin):
    if np.isneginf(ci[0]) or ci[0]<np.min(data):
        ci[0] = np.min(data)
    if np.isinf(ci[1]) or ci[1]>np.max(data):
        ci[1] = np.max(data)
        
    b,stepsize = np.linspace(ci[0],ci[1],num=nbin,retstep=True)
    
    if not(ci[0]==np.min(data)):
        nstep = np.ceil(abs((b[0]-np.min(np.mean(data)))/stepsize)).astype(int)
        L_append = np.linspace(b[0]-stepsize*nstep,b[0]-stepsize,num=nstep-1)
    else:
        L_append = []
    if not (ci[1]==np.max(data)):
        nstep = np.ceil(abs((np.max(data)-b[-1])/stepsize)).astype(int)
        R_append = np.linspace(b[-1]+stepsize,b[-1]+stepsize*nstep,num=nstep-1)
    else:
        R_append = []
    b = np.concatenate((L_append,b,R_append))
    return b
    
def test(data1,data2,nbs=1):
    # from http://faculty.psy.ohio-state.edu/myung/personal/course/826/bootstrap_hypo.pdf
    n = len(data1)
    
    # merge the two samples (sizes n and m)
    X = np.concatenate((data1,data2))
    
    # draw a bootstrap sample from the merged sample
    X_resample = bootstrap_resample(X,n=nbs)
    X1 = np.asarray([X_resample[j][:n] for j in range(len(X_resample))])
    X2 = np.asarray([X_resample[j][n:] for j in range(len(X_resample))])
    
    # calculate the means 
    m1 = np.mean(X1,axis=1)
    m2 = np.mean(X2,axis=1)

    # calculate test statistic t 
    t = m1 - m2
    
    # calculate p   
    if np.mean(data1) > np.mean(data2):
        p = sum(t>(np.mean(data1)-np.mean(data2)))/len(t)
    else:
        p = sum(t<(np.mean(data1)-np.mean(data2)))/len(t)
            
    return p