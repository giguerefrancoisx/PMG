# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:42:58 2017

@author: giguerf
"""

import scipy.fftpack
import numpy as np
#import pandas


def lowpass(x, cut):
    freq = np.fft.fftfreq(len(x), 1)
    z = scipy.fftpack.rfft(x)
    z1 = (freq<cut)*z
#    z1=np.logical_or(z>top,z<-top)*z
    return scipy.fftpack.irfft(z1)

def lowpass_plot(x, cut):
    freq = np.fft.fftfreq(len(x), 1)
    z = scipy.fftpack.fft(x)
    z1 = (freq<cut)*z
#    z1=np.logical_or(z>cut,z<-cut)*z
    x1 = scipy.fftpack.ifft(z1)
    return x1, z, z1, freq

def window_plot(x, cut, w):
    freq = np.fft.fftfreq(len(x), 1)
    z = scipy.fftpack.fft(x)
    z1 = z-np.logical_and(cut<freq, freq<cut+w)*z
    x1 = scipy.fftpack.ifft(z1)
    return x1, z, z1, freq

def plotfunc(num, cutmax, res, lines, time, x, w):
#    cut = cutmax/res*(res-num)
    cut = cutmax/res*num

#    x1, z, z1, freq = lowpass_plot(x, cut)
    x1, z, z1, freq = window_plot(x, cut, w)

    lines[0].set_data(time, x)
    lines[1].set_data(freq, abs(z))
    lines[2].set_data(freq, abs(z1))
    lines[3].set_data(time, x1)
    lines[4].set_data(time, x-x1)

#    lines = [line0, line1, line2, line3, line4]

    return lines

def animate(cut=0.0033, cutmax=0.02, res=400, w=0.001):
#if 1==1:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import GitHub.COM.openbook as ob
    time, data = ob.openbook('P:/AHEC/SAI/')
#    x = data['10CVEHCG0000ACXD'].loc[:,'TC15-163']-data['10SIMERI00INACXD'].loc[:,'TC15-163']
#    x = data['10CVEHCG0000ACXD'].iloc[:,0]
    x = data['11CHST0000H3ACXC'].loc[:,'TC12-012']

    x1, z, z1, freq = lowpass_plot(x, cut)

    plt.close('all')
    fig, axs = plt.subplots(1,2, figsize=(21,7))

    line0, = axs[0].plot(time, x)
    line1, = axs[1].plot(freq, abs(z))
    line2, = axs[1].plot(freq, abs(z1))
    line3, = axs[0].plot(time, x1)
    line4, = axs[0].plot(time, x-x1)

    lines = [line0, line1, line2, line3, line4]

    axs[0].set_xlim(-0.01,0.3)
    axs[0].set_ylim(-35, 10)
    axs[1].set_xlim(-0.01,0.07)
    return animation.FuncAnimation(fig, plotfunc, res, fargs=(cutmax, res, lines, time, x, w),
                                       interval=50, blit=True)
#ani.save('P:/AHEC/Plots/test/ani.htm')
#%%
#    plt.close('all')
#    fig, axs = plt.subplots(1,2, figsize=(21,7))
#
#    x = data['11CHST0000H3ACXC'].loc[:,'TC12-012']
#
#    freq = np.fft.fftfreq(len(x), 1)
#    z = scipy.fftpack.fft(x)
#    f = np.vectorize(lambda t: -8889*t+326 if (t>=0.0085 and t<=0.031) else 20000)
#    z1 = z*(f(freq)>abs(z))
##    z1 = np.logical_and(0.009<freq, freq<0.016)*z
##    z1 = np.logical_and(.1<abs(zr), abs(zr)<1000)*z1
#    x1 = scipy.fftpack.ifft(z1)
#
#    line0, = axs[0].plot(time, x)
#    line1, = axs[1].semilogy(freq, abs(z))
#    line2, = axs[1].semilogy(freq, abs(z1))
#    line3, = axs[0].plot(time, x1)
#    line4, = axs[0].plot(time, x-x1)
##    axs[1].plot([0.0085,0.031], [250,50])
#    axs[1].plot(freq, f(freq))
#
#    axs[0].set_xlim(-0.01,0.3)
#    axs[0].set_ylim(-35, 10)
#    axs[1].set_xlim(-0.01,0.07)
