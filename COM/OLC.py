# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:02:17 2019

@author: giguerf
"""
from PMG.COM.mme import MMEData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz


def olc(vehicle_acc, time=None, idx_stop=None, return_olc_info=False):
    if time is None:
        time = np.arange(0, len(vehicle_acc)*0.0001, 0.0001)
    if idx_stop is None:
        idx_stop = 730
    if not isinstance(vehicle_acc, np.ndarray):
        try:
            vehicle_acc = np.array(vehicle_acc)
        except: 
            raise Exception('Error with vehicle acceleration array')
    if not isinstance(time, np.ndarray):
        try:
            time = np.array(time)
        except: 
            raise Exception('Error with time array')
    dt = time[1]-time[0]
    i0 = np.abs(time).argmin()
        
    vehicle_vel = cumtrapz(vehicle_acc, x=time, initial=0)
    vehicle_vel = (vehicle_vel-vehicle_vel[idx_stop])*9.8067 # in m/s
    vehicle_dis = cumtrapz(vehicle_vel, x=time, initial=0)
    vehicle_dis = (vehicle_dis-vehicle_dis[i0]) # in m

    v_0 = vehicle_vel[i0] #m/s

    occupant_dis_phase_1 = v_0*time # m
    index1 = np.abs(occupant_dis_phase_1-vehicle_dis-0.065).argmin() #Find where the difference is 65 mm
    t1 = time[index1] #Find the time at that point (s)

    index2 = index1 + np.abs(((v_0+vehicle_vel)/2*(time-t1)-(vehicle_dis-vehicle_dis[index1])-0.235)[index1:]).argmin() #Find where the difference is 235 mm
    t2 = time[index2] #Find the time at that point (s)
    v_f = vehicle_vel[index2]
    occupant_dis_phase_2 = v_0*time-(v_0-v_f)/2*(time-t1)**2/(t2-t1)
    
    occupant_dis = np.concatenate((occupant_dis_phase_1[:index1],
                                   occupant_dis_phase_2[index1:index2+1],
                                   occupant_dis_phase_2[index2+1:]*0+float('nan')))


    occupant_vel = np.diff(occupant_dis)/dt #in m/s
    OLC = (v_0-v_f)/(t2-t1)/9.81 # g
    
    
    if return_olc_info:
        return {'OLC': OLC,
                't1': t1,
                't2': t2,
                'vf': v_f,
                'dx_at_t1': occupant_dis[index1]-vehicle_dis[index1],
                'dx_at_t2': occupant_dis[index2-1]-vehicle_dis[index2-1]}
    else:
        return OLC




#test = MMEData('P:/2019/19-6000/19-6008 (208-212-301F)/01 - TC19-109_TOYOTA_HIGHLANDER (Noir)/PMG/TC19-109/TC19-109.mme') #Raw data
#simele = test.channels_from_list(['10SIMELE00INACXD'])[0]
#
##df = pd.DataFrame(data=[simele.time_ms,simele.data]).transpose()
##df.columns=['Time','SIMELE']
#time = pd.Series(simele.time)
#time_ms = pd.Series(simele.time_ms)
#dt = time.iloc[1]-time.iloc[0]
#t0 = time.abs().argmin()
#%%
#idx_stop = 730
##idx_stop = 1000

#vehicle_acc = pd.Series(simele.data)#.rolling(100, min_periods=0).mean() #g
#vehicle_vel = pd.Series(cumtrapz(vehicle_acc, x=time, initial=0))
#vehicle_vel = (vehicle_vel-vehicle_vel.iloc[idx_stop])*9.8067 # in m/s
#vehicle_dis = pd.Series(cumtrapz(vehicle_vel, x=time, initial=0))
#vehicle_dis = (vehicle_dis-vehicle_dis.iloc[t0]) # in m
#
#v_0 = vehicle_vel.iloc[t0] #m/s
#
#occupant_dis_phase_1 = v_0*time # m
#index1 = (occupant_dis_phase_1-vehicle_dis-0.065).abs().idxmin() #Find where the difference is 65 mm
#t1 = time.iloc[index1] #Find the time at that point (s)
##occupant_dis = occupant_dis_phase_1.iloc[:index].append(pd.Series(occupant_dis_phase_1.iloc[index:])*0)
#
#index2 = ((v_0+vehicle_vel)/2*(time-t1)-(vehicle_dis-vehicle_dis[index1])-0.235)[index1:].abs().idxmin() #Find where the difference is 300 mm
#t2 = time.iloc[index2] #Find the time at that point (s)
#v_f = vehicle_vel[index2]
#occupant_dis_phase_2 = v_0*time-(v_0-v_f)/2*(time-t1)**2/(t2-t1)
#
#occupant_dis = occupant_dis_phase_1.iloc[:index1].append(occupant_dis_phase_2.iloc[index1:index2]).append(occupant_dis_phase_2.iloc[index2:]*0+float('nan'))
#
#occupant_vel = occupant_dis.diff()/dt #in m/s
#
#OLC = (v_0-v_f)/(t2-t1)/9.81 # g

#print(f'max: {vehicle_vel.max()*3.6:.2f} km/h\nmin: {vehicle_vel[2100]*3.6:.2f} km/h\nOLC: {OLC:.2f}\nt1: {t1*1000:.2f}, t2: {t2*1000:.2f}\nt(v=0): {time[idx_stop]*1000:.2f}')
#
##% Calculate
#sl = slice(0,2100)
#plt.close('all')
##plt.plot(time_ms, vehicle_acc)
#plt.plot(time_ms[sl], vehicle_vel[sl]*3.6)
#plt.plot(time_ms[sl], occupant_vel[sl]*3.6)
#
#plt.figure()
#plt.plot(time_ms[sl], vehicle_dis[sl])
#plt.plot(time_ms[sl], occupant_dis[sl])
#plt.plot(time_ms[sl], occupant_dis[sl]-vehicle_dis[sl])
#plt.scatter(t1*1000, 0.065)
#plt.scatter(t2*1000, 0.3)

#%%
#plt.figure()
#plt.plot(time_ms[sl], occupant_dis_phase_1[sl])
#plt.plot(time_ms[sl], occupant_dis_phase_2[sl])
#plt.plot(time_ms[sl], occupant_dis[sl])
#%% X-Crash
#index2 = 973
#t2 = time.iloc[index2]
#occupant_dis_phase_2 = v_0*(time-1/2*(time-t1)**2/(t2-t1))
#occupant_dis = occupant_dis_phase_1.iloc[:index1].append(occupant_dis_phase_2.iloc[index1:index2]).append(occupant_dis_phase_2.iloc[index2:]*0+occupant_dis_phase_2.iloc[index2])
#plt.figure()
#plt.plot(time_ms[sl], vehicle_dis[sl])
#plt.plot(time_ms[sl], occupant_dis[sl])
#plt.plot(time_ms[sl], occupant_dis[sl]-vehicle_dis[sl])
#plt.scatter(t1*1000, 0.065)
#plt.scatter(t2*1000, 0.3)
