# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:55:37 2018

@author: giguerf
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PMG.COM import data as dat

time, data = dat.import_data('P:/AHEC/Data/THOR/', '11PELV0000THACXA', check=False)
#time, data = dat.import_data('P:/SLED/Data/', '12PELV0000Y7ACXA', check=False)

#sl = slice(100,1600)
sl = slice(100,1100)
#%%

x = np.linspace((sl.start-100)/10000,(sl.stop-100)/10000,sl.stop-sl.start)

A = 0.11
B = 0.08
C = -28
D = 1870
a = -2
b = 10000
c = 0.04


E = 1/(2*D*(A-B))

x0 = np.array([A,B,C,D,a,b,c])
xl = np.array([0,0,-60,1,-40,-np.inf,0.01])
xh = np.array([0.15,0.15,0,100000,10,np.inf,0.07])

y = C*(x-A)/(B-A)*np.exp(D*(E**2-(x-B-E)**2))

def f(coeffs, x, Y):
    A,B,C,D,a,b,c = coeffs
    ideal = C*(x-A)/(B-A)*np.exp(D*((1/(2*D*(A-B)))**2-(x-B-(1/(2*D*(A-B))))**2))
    knee = -a*np.exp(-b*(x-c)**2)
    return ideal+knee-Y

plt.close('all')
for i in range(20,30):

    plt.figure()
    Y = np.array(data.iloc[sl,i].rolling(100,0,center=True,win_type='triang').mean())
    plt.plot(time[sl], Y)
    plt.plot(time[sl], np.array(data.iloc[sl,i]),color='tab:blue')

    res = scipy.optimize.least_squares(f, x0, args=(x,Y), bounds=(xl,xh))
    A,B,C,D,a,b,c = res.x
    E = 1/(2*D*(A-B))
    y = C*(x-A)/(B-A)*np.exp(D*(E**2-(x-B-E)**2))
    knee = -a*np.exp(-b*(x-c)**2)

    plt.plot(x, y+knee)

    plt.xlim(0,0.15)
    plt.ylim(-80,10)
