# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:02:17 2019

ref: Fig.1 of ESV paper 13-0442
Parametric study using the OLC and spull to qualify the severity of the full width rigid test and design an improved front-end
christophe picquet, richard zeitouni, celine adalian

@author: giguerf
"""
from PMG.COM.mme import MMEData
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import numpy as np

#test = MMEData('P:\\2019\\19-6000\\19-6008 (208-212-301F)\\11 - TC18-173_BMW_X2 (Noir)\\PMG\\TC18-173\\TC18-173.mme') #Raw data
test = MMEData('P:\\2019\\19-6000\\19-6008 (208-212-301F)\\01 - TC19-109_TOYOTA_HIGHLANDER (Noir)\\PMG\\TC19-109\\TC19-109.mme') #Raw data


simele = test.channels[0]

##df = pd.DataFrame(data=[simele.time_ms,simele.data]).transpose()
##df.columns=['Time','SIMELE']
time = pd.Series(simele.time)
#time_ms = pd.Series(simele.time_ms)
#dt = time_ms.iloc[1]-time_ms.iloc[0]
#t0 = time.abs().argmin()
#%%
vehicle_acc = pd.Series(simele.data)#.rolling(100, min_periods=0).mean() #g
#vehicle_vel = pd.Series(cumtrapz(vehicle_acc, x=time))
#vehicle_vel = vehicle_vel.append(pd.Series(vehicle_vel.iloc[-1],index=[4099]))
#vehicle_vel = (vehicle_vel-vehicle_vel.iloc[-1])*9.8067#*3.6 # g*s/35.304 = km/h [m/s]
#vehicle_dis = pd.Series(cumtrapz(vehicle_vel, x=time))
#vehicle_dis = vehicle_dis.append(pd.Series(vehicle_dis.iloc[-1],index=[4099]))
#vehicle_dis = (vehicle_dis-vehicle_dis.iloc[t0])#/3.6*1000 #km/h*s/277.777 = mm [m]

#%% 

def olc(vehicle_acc, time=None, fmt='g'):
    if time is None: 
        time = np.arange(0, len(vehicle_acc)/10000, 0.0001)
        i0 = 0
    else:
        i0 = time.abs().idxmin()
    
    if not isinstance(vehicle_acc, np.ndarray):
        try:
            vehicle_acc = np.array(vehicle_acc)
        except:
            raise Exception('Error in OLC calculation: data is not in the right format!')    
    
    if not isinstance(time, np.ndarray):
        try:
            time = np.array(time)
        except:
            raise Exception('Error in OLC calculation: time is not in the right format!')
            
    vehicle_vel = cumtrapz(vehicle_acc, x=time)
    vehicle_vel = np.append(vehicle_vel, vehicle_vel[-1]) # make acceleration, velocity, and displacement all have the same length
    
    
    vehicle_vel = vehicle_vel*9.8067 
    vehicle_vel = vehicle_vel - (vehicle_vel[0]-56/3.6)
    
    
#    vehicle_vel = (vehicle_vel-vehicle_vel[-1])*9.8067
    vehicle_dis = cumtrapz(vehicle_vel, x=time)
    vehicle_dis = np.append(vehicle_dis, vehicle_dis[-1])
    vehicle_dis = vehicle_dis-vehicle_dis[i0]
    
    vehicle_acc = vehicle_acc[time>=0]
    vehicle_vel = vehicle_vel[time>=0]
    vehicle_dis = vehicle_dis[time>=0]
    time = time[time>=0]
    v0 = vehicle_vel[0]
    
    # find t1
    occupant_dis = v0*time
    delta = occupant_dis - vehicle_dis
    i1 = np.where(delta>0.065)[0][0]
    t1 = time[i1]
    
    # find t2 using the area of a trapezoid
    delta_theo = delta
    delta_theo[i1:] = (0.5*(t1 + time)*v0 - vehicle_dis)[i1:]
    i2 = np.where(delta>0.235)[0][0]
    t2 = time[i2]
    
    olc = v0/(t2-t1)
    if fmt=='g':
        return olc/9.8067, t1, t2, i1, i2, v0, vehicle_vel, vehicle_dis
    else:
        return olc

# uses pd.Series whereas the above uses np.arrays
#def olc2(vehicle_acc, time=time, fmt='g'):
#    if time is None:
#        time = pd.Series(np.arange(0, len(vehicle_acc)/10000, 0.0001))
#        t0 = 0
#    else:
#        t0 = time.abs().idxmin()
#    vehicle_acc = pd.Series(vehicle_acc)
#    vehicle_vel = pd.Series(cumtrapz(vehicle_acc, x=time))
#    vehicle_vel = vehicle_vel.append(pd.Series(vehicle_vel.iloc[-1],index=[4099]))
#    vehicle_vel = (vehicle_vel-vehicle_vel.iloc[-1])*9.8067#*3.6 # g*s/35.304 = km/h [m/s]
#    vehicle_dis = pd.Series(cumtrapz(vehicle_vel, x=time))
#    vehicle_dis = vehicle_dis.append(pd.Series(vehicle_dis.iloc[-1],index=[4099]))
#    vehicle_dis = (vehicle_dis-vehicle_dis.iloc[t0])#/3.6*1000 #km/h*s/277.777 = mm [m]
#    
#    vehicle_acc = vehicle_acc[time>=0].reset_index(drop=True)
#    vehicle_vel = vehicle_vel[time>=0].reset_index(drop=True)
#    vehicle_dis = vehicle_dis[time>=0].reset_index(drop=True)
#    time = time[time>=0].reset_index(drop=True)
#    
#    v0 = vehicle_vel[0]
#    
#    # find t1
#    occupant_dis = v0*time
#    delta = occupant_dis - vehicle_dis
#    i1 = delta[delta<=0.065].index[-1]
#    t1 = time.iloc[i1]
#    
#    # find t2 using the area of a trapezoid
#    delta_theo = delta
#    delta_theo[i1:] = (0.5*(t1 + time)*v0-vehicle_dis)[i1:]
#    i2 = delta[delta<=0.235].index[-1]
#    t2 = time.iloc[i2]
#    
#    olc = v0/(t2-t1)
#    if fmt=='g':
#        return olc/9.8067
#    else:
#        return olc


#print(olc(vehicle_acc, time=time))
#print(olc2(vehicle_acc, time=time))
# extra stuff to get occupant velocities and displacements
        
olc, t2, t2, i1, i2, v0, vehicle_vel, vehicle_dis = olc(vehicle_acc)

occupant_vel = np.append(np.repeat(v0, i1-1), np.linspace(v0, 0, i2-i1+1))
occupant_dis = pd.Series(cumtrapz(occupant_vel, x=time[:len(occupant_vel)]))
delta = occupant_dis - vehicle_dis[:len(occupant_dis)]



fig, ax = plt.subplots()
ax.plot(time, vehicle_dis, label='vehicle')
ax.plot(time[:len(occupant_dis)], occupant_dis, label='occupant')
ax.plot(time[:len(delta)], delta, label='difference')
ax.scatter(t1, 0.065)
ax.scatter(t2, 0.235)
ax.legend()
ax.set_title('Displacement')
#

fig, ax = plt.subplots()
ax.plot(time, vehicle_vel, label='vehicle')
ax.plot(time[:len(occupant_vel)], occupant_vel, label='occupant')
ax.axhline(0, color='k')
ax.axvline(0, color='k')
ax.legend()
ax.set_title('Velocity')
