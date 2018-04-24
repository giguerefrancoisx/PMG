# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:14:43 2018

@author: giguerf
"""

import matplotlib.pyplot as plt
from PMG.COM.openbook import openHDF5
from PMG.COM import table as tb
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

table = tb.get('SLED')

SLED = 'P:/SLED/Data/'
chlist = ['12HEAD0000Y7ACXA','12HEAD0000Y2ACXA',
          '12CHST0000Y7ACXC','12CHST0000Y2ACXC',
          '12PELV0000Y7ACXA','12PELV0000Y2ACXA']

time, fulldata = openHDF5(SLED, chlist)

raw = fulldata['12PELV0000Y7ACXA'].iloc[:,0]

#%% Animation
fig, ax = plt.subplots(figsize=(5, 3))
#ax.set(xlim=(-3, 3), ylim=(-1, 1))
ax.plot(raw, 'k')

windows = range(0,101,2)

F = [0]*len(windows)
for i , w in enumerate(windows):
    F[i] = raw.rolling(w+1,0,center=True,win_type='parzen').mean()
line = ax.plot(F[0], color='b', lw=2)[0]
text = ax.text(0.1,0.1,'0',transform=ax.transAxes)

def run_animation():
    anim_running = True

    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    def animFunc(i):
        line.set_ydata(F[i])
        text.set_text(windows[i])

    fig.canvas.mpl_connect('button_press_event', onClick)

    anim = FuncAnimation(fig, animFunc, interval=100, frames=len(F))

run_animation()

#anim.save('filename.mp4')
#anim.save('filename.gif', writer='imagemagick')
#%% Slider
plt.close('all')
fig, ax = plt.subplots(figsize=(5,3))
gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
#ax.set(xlim=(-3, 3), ylim=(-1, 1))
ax1.plot(raw, 'k')

slider1 = Slider(ax2, 'Window', 0, 200, valinit=30, valfmt='%1.0f')
line = ax1.plot(raw.rolling(slider1.valinit+1,0,center=True,win_type='parzen').mean(), color='b', lw=2)[0]
#line2 = ax1.plot(raw.rolling(slider1.valinit+1,0,center=True,win_type='triang').mean(), color='g', lw=2)[0]
#line3 = ax1.plot(raw.rolling(slider1.valinit+1,0,center=True,win_type='boxcar').mean(), color='r', lw=2)[0]
text = ax1.text(0.1,0.1,'30',transform=ax1.transAxes)


def update(val):
    w = int(round(val/2)*2)
    line.set_ydata(raw.rolling(w+1,0,center=True,win_type='parzen').mean())
#    line2.set_ydata(raw.rolling(w+1,0,center=True,win_type='triang').mean())
#    line3.set_ydata(raw.rolling(w+1,0,center=True,win_type='boxcar').mean())
    text.set_text(w)

slider1.on_changed(update)
plt.show()

#%% Slider 2
import scipy.signal as signal
plt.close('all')
fig, ax = plt.subplots(figsize=(5,3))
gs = gridspec.GridSpec(3, 1, height_ratios=[4,1,1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
#ax.set(xlim=(-3, 3), ylim=(-1, 1))
ax1.plot(raw, 'k')

slider1 = Slider(ax2, 'Window1', 0, 200, valinit=30, valfmt='%1.0f')
slider2 = Slider(ax3, 'Window2', 0, 200, valinit=30, valfmt='%1.0f')
line = ax1.plot(raw.rolling(slider1.valinit+1,0,center=True,win_type='parzen').mean(), color='b', lw=2)[0]
line2 = ax1.plot(signal.savgol_filter(raw,slider2.valinit+1,3), color='g', lw=2)[0]
#line3 = ax1.plot(raw.rolling(slider1.valinit+1,0,center=True,win_type='boxcar').mean(), color='r', lw=2)[0]
text = ax1.text(0.1,0.1,'30',transform=ax1.transAxes)


def update(val):
    w = int(round(val/2)*2)
    line.set_ydata(raw.rolling(w+1,0,center=True,win_type='parzen').mean())
#    line2.set_ydata(raw.rolling(w+1,0,center=True,win_type='parzen').mean().rolling(w+1,0,center=True,win_type='parzen').mean())
#    line3.set_ydata(raw.rolling(w+1,0,center=True,win_type='boxcar').mean())
    text.set_text(w)

def update2(val):
    w = int(round(val/2)*2)
    line2.set_ydata(signal.medfilt(raw,w+1))

slider1.on_changed(update)
slider2.on_changed(update2)
plt.show()
