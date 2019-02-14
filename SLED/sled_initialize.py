# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:09:55 2019
Initializing values
@author: tangk
"""
import json
import pandas as pd
import numpy as np
from PMG.read_data import initialize, get_se_angle
from PMG.COM.get_props import *

directory = 'P:\\Data Analysis\\Projects\\SLED\\'
cutoff = range(100,1600)

channels = ['S0SLED000000ACXD',
            '12HEAD0000Y2ACXA',
            '12HEAD0000Y2ACYA',
            '12HEAD0000Y2ACZA',
            '12HEAD0000Y2ACRA',
            '12CHST0000Y2ACXC',
            '12CHST0000Y2ACYC',
            '12CHST0000Y2ACZC',
            '12CHST0000Y2ACRC',
            '12HEAD0000Y7ACXA',
            '12HEAD0000Y7ACYA',
            '12HEAD0000Y7ACZA',
            '12HEAD0000Y7ACRA',
            '12NECKUP00Y7FOXA',
            '12NECKUP00Y7FOZA',
            '12NECKUP00Y7FORA',
            '12NECKUP00Y7MOXB',
            '12CHST0000Y7ACXC',
            '12CHST0000Y7ACYC',
            '12CHST0000Y7ACZC',
            '12CHST0000Y7ACRC',
            '12CHST0000Y7DSXB',
            '12LUSP0000Y7FOXA',
            '12LUSP0000Y7FOYA',
            '12LUSP0000Y7FOZA',
            '12LUSP0000Y7MOXA',
            '12LUSP0000Y7MOYA',
            '12LUSP0000Y7MOZA',
            '12PELV0000Y7ACXA',
            '12PELV0000Y7ACZA',
            '12PELV0000Y7ACRA',
            '12SEBE0000B3FO0D',
            '12SEBE0000B6FO0D']

query_list = [['DUMMY',['Y2','Y7']],
              ['INSTALL',['H1','H3','HB','LB','C1','B0','B11','B12']]]

exclude = ['SE16-0343', # <- this is Summit H3
           'SE16-0395',
           'SE16-0399',
           'SE17-0018_2']

table, t, chdata = initialize(directory, 
                              channels,
                              cutoff,
                              drop=exclude,
                              query_list=query_list)
append = [chdata]

#%% angle stuff
displacement = get_se_angle(chdata.index)
se_angles = displacement.apply(get_angle, axis=1).apply(lambda x: x-x[0]).rename('Angle')
abs_angles = displacement.apply(get_angle, axis=1).rename('Abs_Angle')
angle_t = displacement['Time']/1000
position = displacement.drop('Time', axis=1)
displacement = position.applymap(lambda x: x-x[0]).rename(lambda x: 'D'+x, axis=1)


# shift certain signals
displacement.loc[['SE16-0378','SE16-0381','SE16-0379']] = displacement.loc[['SE16-0378','SE16-0381','SE16-0379']].applymap(lambda x: x[75:])
se_angles.loc[['SE16-0378','SE16-0381','SE16-0379']] = se_angles.loc[['SE16-0378','SE16-0381','SE16-0379']].apply(lambda x: x[75:])
angle_t.loc[['SE16-0378','SE16-0381','SE16-0379']] = angle_t.loc[['SE16-0378','SE16-0381','SE16-0379']].apply(lambda x: x[:-75])

append.append(displacement)
append.append(se_angles)
append.append(abs_angles)
append.append(position)
#%% clean up data
set_to_nan = [['SE16-0414','12PELV0000Y7ACRA'],
              ['SE16-0337','12LUSP0000Y7FOXA'],
              ['SE16-0342','12LUSP0000Y7FOXA'],
              ['SE16-0340','12LUSP0000Y7FOXA']]

for i in set_to_nan:
    chdata.at[i[0],i[1]] = np.repeat(np.nan,len(cutoff))
    
chdata.at['SE16-0338_2','12CHST0000Y7DSXB'] = -chdata.at['SE16-0338_2','12CHST0000Y7DSXB']

#%% preprocessing
append.append((chdata['12CHST0000Y7ACXC']-chdata['12PELV0000Y7ACXA'].apply(smooth_data)).rename('Chest-Pelvis'))

#%% append columns and replace nan with array of nan
chdata = pd.concat(append,axis=1)
chdata = chdata.applymap(lambda x: np.repeat(np.nan,len(cutoff)) if type(x)==np.float else x)
#%% get peaks, etc 
# fix this   
def get_all_features(csv_write=False,json_write=False):                    
    i_to_t = get_i_to_t(t)
    feature_funs = {'Min_': [get_min],
                    'Max_': [get_max],
                    'Tmin_': [get_argmin,i_to_t],
                    'Tmax_': [get_argmax,i_to_t]}
    
    # peak and time to peak of each channel
    features = pd.concat(chdata.chdata.get_features(feature_funs).values(),axis=1,sort=True)
    
    # append other features: component channels at peak resultant channel
    append = []
    for ch in ['12HEAD0000Y2ACRA','12CHST0000Y2ACRC','12HEAD0000Y7ACRA','12CHST0000Y7ACRC']:
        component_channels = [ch[:14] + ax + ch[-1] for ax in ['X','Y','Z']]
        
        component_at_res = get_x_at_y(chdata[component_channels], chdata[ch].apply(get_argmax)).abs()
        append.append(component_at_res.rename(lambda x: 'X' + x + '_at_' + ch, axis=1))
        append.append(component_at_res.div(chdata[ch].apply(get_max),axis=0).rename(lambda x: x + '/' + ch, axis=1).abs())
    
    # get displacements at peak angle change
    d_at_peak_angle = get_x_at_y(displacement, se_angles.apply(get_argmax))
    append.append(d_at_peak_angle.rename(lambda x: x + '_at_Angle', axis=1))
    
    # get sled acceleration at peak chest Acx
    sled_at_peak_chest = get_x_at_y(chdata['S0SLED000000ACXD'], chdata['12CHST0000Y7ACXC'].apply(get_argmin))
    append.append(sled_at_peak_chest.rename('S0SLED000000ACXD_at_12CHST0000Y7ACXC'))
    
    #sled at peak chest acr
    sled_at_peak_chest = get_x_at_y(chdata['S0SLED000000ACXD'], chdata['12CHST0000Y7ACRC'].apply(get_argmax))
    append.append(sled_at_peak_chest.rename('S0SLED000000ACXD_at_12CHST0000Y7ACRC'))
    
    chest_pelvis_at_chest_dx = get_x_at_y(chdata['Chest-Pelvis'], chdata['12CHST0000Y7DSXB'].apply(get_argmin))
    append.append(chest_pelvis_at_chest_dx.rename('Chest-Pelvis_at_12CHST0000Y7DSXB'))
    
    chest_pelvis_at_chest = get_x_at_y(chdata['Chest-Pelvis'], chdata['12CHST0000Y7ACRC'].apply(get_argmax))
    append.append(chest_pelvis_at_chest.rename('Chest-Pelvis_at_12CHST0000Y7ACRC'))
    
    # get time between onset and peak angle
    append.append((se_angles.apply(get_argmax)-se_angles.apply(get_onset_to_max)).rename('TRise_Angle'))
    
    # get excursions and angle changes at peak DDown_y
    append.append(get_x_at_y(chdata[['Angle','DUp_x','DUp_y','DDown_x']],chdata['DDown_y'].apply(get_argmin)).rename(lambda x: x + '_at_DDown_y',axis=1))
    
    # get difference in time to peak of DDown_y and Angle
    append.append((chdata['DDown_y'].apply(get_argmin)-chdata['Angle'].apply(get_argmax)).rename('TDDown_y-Angle'))
    append.append((chdata['DDown_x'].apply(get_argmin)-chdata['Angle'].apply(get_argmax)).rename('TDDown_x-Angle'))
    append.append((chdata['DDown_x'].apply(get_argmin)-chdata['DDown_y'].apply(get_argmax)).rename('TDDown_x-DDown_y'))
    
    #
    sb_at_min_chestz = get_x_at_y(chdata['12SEBE0000B3FO0D'], chdata['12CHST0000Y2ACZC'].apply(get_argmin))
    append.append(sb_at_min_chestz.rename('12SEBE0000B3FO0D_at_Min_12CHST0000Y2ACZC'))
    sb_at_min_headz = get_x_at_y(chdata['12SEBE0000B3FO0D'], chdata['12HEAD0000Y2ACZA'].apply(get_argmin))
    append.append(sb_at_min_headz.rename('12SEBE0000B3FO0D_at_Min_12HEAD0000Y2ACZA'))
    
    # head and chest 3ms, head and knee excursions
    append.extend([table[['Head 3ms', 'Chest 3ms', 'Head Excursion', 'Knee Excursion']].rename(lambda x: x.replace(' ','_'),axis=1).rename_axis('SE'),
                   (features['Min_12CHST0000Y7ACXC']-features['Min_12PELV0000Y7ACXA']).rename('Chest-Pelvis'),
                   (features['Tmin_12PELV0000Y7ACZA']-features['Tmin_12PELV0000Y7ACXA']).rename('t_Chest-Pelvis')])
    
    features = pd.concat([features]+append,axis=1,sort=True)

    if csv_write:
        features.to_csv(directory + 'features.csv')
        
    if json_write:
        to_JSON = {'project_name': 'FMVSS213_sled_comparison',
                   'directory'   : directory}
        
        with open(directory+'params.json','w') as json_file:
            json.dump(to_JSON,json_file)
    
    return features

