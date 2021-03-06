# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:44:50 2018

@author: tangk
"""
import pandas as pd
import numpy as np
import copy
import re
import matplotlib.pyplot as plt
from PMG.COM.plotfuns import *
from PMG.COM.easyname import get_units, rename
from PMG.COM.helper import *
from PMG.COM.arrange import *
import json
from PMG.SLED.sled_initialize import table, t, chdata, angle_t, get_all_features
from PMG.SLED.sled_helper import *
from functools import partial
import glob
import copy

directory = 'P:\\Data Analysis\\Projects\\SLED\\'
#%% plotting params
plot_specs = {'new_accel': {'color': 'IndianRed'},
              'old_accel': {'color': 'SkyBlue'},
              'new_decel': {'color': '#589F58'},
              'B11'      : {'linestyle': '-'},
              'B12'      : {'linestyle': '--'}}
comparison = ['old_accel','new_decel']
dummies = ['Y7','Y2']
names = pd.read_csv(directory+'names.csv',index_col=0,header=None,squeeze=True).apply(lambda x: x.replace('\r',''))
rename = partial(rename, names=names)
install_order = {'Y7': ['H1','H3','HB','LB'],
                 'Y2': ['C1','B0','B11','B12']}

models = ['Evenflo Embrace  CRABI','SNUGRIDE CRABI','Cybex Aton Q']
thresholds = {'Head_Excursion': [720, 813],
              'Knee_Excursion': [915],
              'Head_3ms': [80],
              'Chest_3ms': [60]}
linestyles = ['--',':']
#%%
features = get_all_features(csv_write=False,json_write=False)

#%%
with open(directory+'params.json','r') as json_file:
    to_JSON = json.load(json_file)

#pd.DataFrame(to_JSON['res']['Y7']['bench']['p']).set_index('_row',drop=True)

#%% find which comparisons have p<0.05 and beta>threshold 
# fix this
def get_threshold(threshold,response):
    for th in threshold.keys():
        if th in response:
            return threshold[th]
    return 0

threshold = {'Tmax': 0.005,
             'Tmin': 0.005}

drop_responses = ['Max_12HEAD0000Y7ACXA',
                  'Min_12HEAD0000Y7ACZA',
                  'Min_12HEAD0000Y7ACRA',
                  'Min_12NECKUP00Y7FOZA',
                  'Min_12PELV0000Y7ACRA',
                  'Max_12NECKUP00Y7FOXA',
                  'Max_12CHST0000Y7ACXC',
                  'Max_12CHST0000Y7ACZC',
                  'Min_12CHST0000Y7ACRC',
                  'Max_12CHST0000Y7DSXB',
                  'Max_12PELV0000Y7ACXA',
                  'Max_12PELV0000Y7ACZA',
                  'Min_12SEBE0000B3FO0D',
                  'Min_12SEBE0000B6FO0D',
                  'Min_12HEAD0000Y2ACXA',
                  'Min_12HEAD0000Y2ACZA',
                  'Min_12HEAD0000Y2ACRA',
                  'Min_12CHST0000Y2ACXC',
                  'Min_12CHST0000Y2ACRC',
                  'Min_12PELV0000Y2ACXA',
                  'Min_12PELV0000Y2ACZA',
                  'Min_Angle',
                  'Max_DUp_x',
                  'Max_DUp_y',
                  'Max_DDown_x',
                  'Max_DDown_y']


res = pd.DataFrame([],columns=['Dummy','Comparison','Install','Response','p','beta'])
for d in to_JSON['res'].keys():
    for c in to_JSON['res'][d].keys():
        p = pd.DataFrame(to_JSON['res'][d][c]['p']).set_index('_row',drop=True)   
        beta = pd.DataFrame(to_JSON['res'][d][c]['beta']).set_index('_row',drop=True) 
        for i in p.index:
            is_sig = p.loc[i]<0.1
            ch_sig = p.columns[is_sig]
            
            drop = ch_sig.intersection(drop_responses).union(ch_sig.intersection(['Tm' + s[1:] for s in drop_responses]))
            ch_sig = ch_sig.drop(drop)
            
            beta_sig = beta[ch_sig].loc[i]
            
            df = pd.DataFrame.from_dict({'Dummy'     : [d]*len(ch_sig),
                                         'Comparison': [c]*len(ch_sig),
                                         'Install'   : [i]*len(ch_sig),
                                         'Response'  : ch_sig,
                                         'p'         : p[ch_sig].loc[i].values,
                                         'beta'      : beta_sig.values})
            res = pd.concat((res,df),ignore_index=True)

drop = []
for i in range(len(res)):
    response = res.at[i,'Response']
    beta = abs(res.at[i,'beta'])
    thresh = get_threshold(threshold,response)
    if beta<thresh:
        drop.append(i)
    
res = res.drop(drop,axis=0)

res.to_csv(directory + 'comparison.csv')            
            
#res.append({'Dummy':1,'Comparison':2,'Install':3,'Response':4,'p':5,'beta':6},ignore_index=True)

#%% get mean difference for each pair
# fix this
grp = table.groupby(['MODEL','INSTALL','DUMMY','SLED']).mean()[['Head 3ms','Chest 3ms','Head Excursion','Knee Excursion']]
diff = {i: pd.DataFrame(index=table.groupby(['MODEL','INSTALL','DUMMY']).groups,columns=['Head 3ms','Chest 3ms','Head Excursion','Knee Excursion']) for i in comparison}

for c in comparison:
    for i in diff[c].index:
        if i + (c,) in grp.index:
            diff[c].loc[i] = grp.loc[i+(c,)] - grp.loc[i+('new_accel',)]
    diff[c].index.names = ['MODEL','INSTALL','DUMMY']
#    diff[c].to_csv(directory + c + '.csv')

#%% CATEGORICAL
#%% bar plots comparing different responses on the two benches or on the two sleds
plot_channels = ['Max_12SEBE0000B3FO0D',
                 'Max_12NECKUP00Y7FOZA',
                 'Min_12CHST0000Y7DSXB',
                 'Head_Excursion',
                 'Knee_Excursion',
                 'Head_3ms',
                 'Chest_3ms',
                 'Max_Angle',
                 'Min_DUp_x',
                 'Min_DDown_x',
                 'Min_DUp_y',
                 'Min_DDown_y',
                 'Max_12LUSP0000Y7MOZA',
                 'X12CHST0000Y2ACXC_at_12CHST0000Y2ACRC',
                 'X12CHST0000Y2ACZC_at_12CHST0000Y2ACRC']

grouped = table.groupby(['DUMMY'])
for d in dummies:
    for c in comparison: 
        subset  = grouped.get_group(d).table.query_list('SLED',['new_accel',c])
        pvals = pd.DataFrame(to_JSON['res'][d]['bench' if c=='old_accel' else 'sled']['p']).set_index('_row',drop=True)
        
        for ch in plot_channels:
            x = intersect_columns(arrange_by_group(subset,features[ch],'SLED','INSTALL', col_order=install_order[d]))
            if x==None: continue
            for k in x:
                x[rename(k)] = x.pop(k).abs()
                
            # plot
            fig, ax = plt.subplots()
            ax = plot_bar(ax, x, var=get_var, plot_specs={rename(k): plot_specs[k] for k in plot_specs})
            
            # add significance
            installs = [l.get_text() for l in ax.get_xticklabels()]
            y = np.max([l.get_data()[1] for l in ax.lines],axis=0)
            if ch in pvals:
                p = pvals[ch]
            else:
                p = pd.Series(1, index=installs)
            add_stars(ax, ax.get_xticks(), p[install_order[d]].dropna(), y, fontsize=24)
            
            # set labels and font sizes
            ax = set_labels(ax, {'title': rename(ch), 'ylabel': get_units(ch)})
            ax = adjust_font_sizes(ax, {'ticklabels':22,'title':28,'ylabel':26})
#            ax = adjust_font_sizes(ax, {'ticklabels':20,'title':24,'ylabel':20})
#            ax.legend(ncol=2,bbox_to_anchor=(1,1))
            if ch in thresholds:
                for i in range(len(thresholds[ch])):
                    ax.axhline(thresholds[ch][i], linestyle=linestyles[i], color='r')
                    y = np.append(y, thresholds[ch][i])
            ax.set_ylim([0, 1.2*max(y)])
            ax.set_xticklabels([rename(x) for x in installs])
            plt.show()
            plt.close(fig)

#%% bar plots with model on the x axis
plot_channels = ['energy_trapz',
                 'energy_simpson']
subset = (table.table.query_list('MODEL', ['PRONTO HIII-6-YR'])
                .table.query_list('SLED',['new_accel', 'new_decel']))
subset['MODEL'] = subset['MODEL'].replace(names)

for ch in plot_channels:
    x = arrange_by_group(subset, features[ch], 'SLED','MODEL')
    print(ch)
    print(x)
    fig, ax = plt.subplots()
    ax = plot_bar(ax, x, plot_specs=plot_specs)
    ax = set_labels(ax, {'title': rename(ch), 'ylabel': get_units(ch), 'legend': {'bbox_to_anchor': (1,1)}})
    ax = adjust_font_sizes(ax, {'ticklabels': 20, 'title': 24, 'axlabels': 20})
#%% plot ranges of differences
plot_channels = ['Head 3ms','Chest 3ms','Head Excursion','Knee Excursion']
for c in comparison:
    subset = diff[c].query('INSTALL!=\'B22\'').groupby('DUMMY')
    for grp in subset:
        for ch in plot_channels:
            x = arrange_by_group(grp[1], grp[1][ch], 'INSTALL')
            if x=={}: continue
            fig, ax = plt.subplots()
            ax = plot_range(ax, x, c={i: (0.5, 0.5, 0.5) for i in x}, order=install_order[grp[0]])
            ax.axvline(0, color='k', linestyle='--', linewidth=1.3)
            ax = set_labels(ax, {'title': ch, 'xlabel': get_units(ch)})
            if '3ms' in ch:
                ax.set_xlim(-15,15)
            else:
                ax.set_xlim(-100,100)
            ax = adjust_font_sizes(ax, {'ticklabels': 18, 'title': 24, 'axlabels': 20})
            plt.show()
            plt.close(fig)

#%% euclidean distance of B11 vs B12
installs = ['B11','B12']
cols = ['DUp_x','DUp_y','DDown_x','DDown_y']
grouped = table.table.query_list('MODEL',models).table.query_list('INSTALL',installs).groupby(['MODEL','SLED'])
distances = pd.Series(index=grouped.groups.keys())

# get distances
for grp in grouped:
    subgroup = grp[1].groupby('INSTALL').groups
    distances[grp[0]] = get_distance(chdata.loc[subgroup['B11'], cols], chdata.loc[subgroup['B12'], cols])  
#    distances[grp[0]] = get_distance(chdata.loc[subgroup['B12'],cols], chdata.loc[subgroup['B11'], cols])  
distances = distances.rename(names).unstack().to_dict()

for c in comparison:
    x = {k: pd.DataFrame(distances[k],index=[0]) for k in [rename('new_accel'), rename(c)]}   
    fig, ax = plt.subplots()
    ax = plot_bar(ax, x, errorbar=False, plot_specs={rename(k): copy.deepcopy(plot_specs[k]) for k in plot_specs})
    ax = set_labels(ax, {'ylabel': 'Distance Metric', 'title': '(TBD)','legend': {'bbox_to_anchor': [0.95, 1.4], 'ncol': 2}})
    ax = adjust_font_sizes(ax,{'ticklabels': 20,'title': 24,'legend':20,'ylabel':20})

#%% get euclidean distance between configurations
grouped = (table.query('DUMMY==\'Y2\'')
                .groupby(['INSTALL','MODEL']))

for c in comparison:
    dist = pd.DataFrame()
    for grp in grouped:
        print(grp[0])
        subgroup = grp[1].groupby('SLED').groups
        x = chdata.loc[subgroup['new_accel'], ['DUp_x','DDown_x','DUp_y','DDown_y']]
        if c in subgroup:
            y = chdata.loc[subgroup[c], ['DUp_x','DDown_x','DUp_y','DDown_y']]
        else: 
            continue
        dist.at[grp[0][1], grp[0][0]] = get_distance(x, y, lagfun=get_lag2)
    fig, ax = plt.subplots()
    plot_bar(ax, {'label': dist}, errorbar=True, order=install_order['Y2'], plot_specs = {'label': {'color': (0.65,0.65,0.65)}})
        
#%% TIME SERIES OVERLAYS
#%% plot overlays model-by-model
plot_channels = ['12HEAD0000Y2ACXA',
                 '12HEAD0000Y2ACZA',
                 '12HEAD0000Y2ACRA']
grouped = (table.drop('SE16-0338')
                .table.query_list('INSTALL',['B11', 'B12'])
                .table.query_list('MODEL',models)
                .table.query_list('SLED',['old_accel'])
#                .table.query_list('MODEL',['PRONTO HIII-6-YR'])
                .groupby(['MODEL']))

for ch in plot_channels:
    for subset in grouped:
        subset[1]['SLED'] = subset[1]['SLED'].replace(names)
        x = arrange_by_group(subset[1], chdata[ch], 'INSTALL')
        if len(x)==0: continue
        print(subset[0])
        line_specs = {rename(k): copy.deepcopy(plot_specs[k]) for k in plot_specs}
        line_specs[rename('new_accel')]['alpha'] = 0.3
        line_specs[rename('new_decel')]['alpha'] = 0.3
        fig, ax = plt.subplots()
        ax = plot_overlay(ax, t, x, line_specs=line_specs)
#        ax.plot(t, chdata.at['SE16-0338',ch], color='#ff00bf', linewidth=2, linestyle='--', label='Model F low')
        ax = set_labels(ax, {'title': rename(ch), 'xlabel': 'Time [s]', 'ylabel': get_units(ch), 'legend': {'bbox_to_anchor': (1,1)}})
        ax = adjust_font_sizes(ax,{'ticklabels': 20,'title': 24,'legend':20,'axlabels':20})
        plt.show()
        plt.close(fig)     

#%% plot multiple channels on the same axis
plot_channels = ['12HEAD0000Y2ACXA',
                 '12HEAD0000Y2ACZA',
                 '12HEAD0000Y2ACRA']
grouped = (table.query('DUMMY==\'Y2\'')
                .table.query_list('SLED',['old_accel'])
                .table.query_list('MODEL',models)
                .table.query_list('INSTALL',['B11','B12'])
                .groupby(['INSTALL','SLED','MODEL']))
for grp in grouped:
    fig, ax = plt.subplots()
    for ch in plot_channels:
#        x = {ch[14] if ch[14] in ['X','Y','Z'] else 'Resultant': chdata.loc[grp[1].index, ch]}
        x = {rename(ch): chdata.loc[grp[1].index, ch]}
        if len(x)==0: continue
        ax = plot_overlay(ax,t,x,line_specs={rename(ch): {'linewidth': 2}})
    ax.axhline(0, color='k', linewidth=1)
#    ax.set_yticks(range(-40, 80, 20))
#    ax = set_labels(ax, {'title': rename(grp[0][0]).replace('\n','') + ', ' + rename(grp[0][2]), 'legend': {}, 'xlabel': 'Time [s]', 'ylabel': 'Acceleration [g]'})
    ax = set_labels(ax, {'title': grp[0], 'legend': {'bbox_to_anchor': (1,1)}, 'xlabel': 'Time [s]', 'ylabel': 'Acceleration [g]'})
    ax = adjust_font_sizes(ax,{'ticklabels':18, 'title':20, 'axlabels':18, 'legend':18})
    ax.set_ylim([-35, 90])
    plt.show()
    plt.close(fig)
#%% plot overlays of excursions vs. time comparing installations
plot_channels = ['DUp_x',
                 'DDown_x',
                 'Angle']
models = ['Evenflo Embrace  CRABI','SNUGRIDE CRABI','Cybex Aton Q']
grouped = (table.query('DUMMY==\'Y2\'')
                .table.query_list('MODEL',models)
                .table.query_list('INSTALL',['B11','B12'])
                .query('SLED==\'old_accel\'')
                .groupby(['MODEL']))
line_specs = {'B11': {'linestyle': '-'},
              'B12': {'linestyle': '--'}}

for ch in plot_channels:
    for grp in grouped:
#        for i in line_specs:
#            line_specs[i]['color'] = plot_specs[grp[0][0]]['color']
        # x is time; y is ch 
        x = arrange_by_group(grp[1], angle_t, 'INSTALL')
        y = arrange_by_group(grp[1], chdata[ch], 'INSTALL')
        if x=={} or y=={}: continue 
        fig, ax = plt.subplots()
        ax = plot_overlay_2d(ax, x, y, line_specs=line_specs)
        ax = set_labels(ax, {'title': str(grp[0]) + ' ' + ch, 'xlabel': 'Time', 'ylabel': 'Displacement', 'legend': {}})
        plt.show()
        plt.close(fig)

#%% plot time series one test at a time
subset = (table.query('DUMMY==\'Y2\'')
               .table.query_list('INSTALL',['B11'])
#               .table.query_list('MODEL',models)
               .table.query_list('SLED',['new_accel','new_decel']))

for i in subset.index:
    if i not in angle_t.index:
        continue
    elif np.all(np.isnan(chdata.at[i,'Angle'])):
        continue
    elif np.all(np.isnan(chdata.at[i,'DDown_x'])):
        continue
    fig, ax = plt.subplots()
    ax.plot(angle_t[i], chdata.at[i,'Angle'],label='Angle')
    ax.plot(angle_t[i], -chdata.at[i,'DUp_x']/10,label='DUp_x')
    ax.plot(angle_t[i], -chdata.at[i,'DUp_y']/10,label='DUp_y')
    ax.plot(angle_t[i], -chdata.at[i,'DDown_x']/10,label='DDown_x')
    ax.plot(angle_t[i], -chdata.at[i,'DDown_y']/10,label='DDown_y')
    ax.set_xticks(np.linspace(0,0.14,8))
    ax.set_yticks(range(0,20,5))
    ax.axhline(1,color='k',linewidth=1)
    ax.legend()
    ax.set_title(table.at[i,'MODEL'] + ' ' + table.at[i, 'SLED'] + ' ' + table.at[i,'INSTALL'] + ' ' + str(i))
    plt.show()

#%%
grouped = (table.query('DUMMY==\'Y2\'')
               .table.query_list('INSTALL',['B11','B12'])
               .table.query_list('MODEL',models)
               .table.query_list('SLED',['old_accel'])
               .groupby('MODEL'))

for grp in grouped:
    subset = grp[1].groupby('INSTALL').groups
    fig, ax = plt.subplots()
    for i in ['B11','B12']:
#        y = {j: -chdata.loc[subset[i], j]/10 for j in ['DUp_x','DUp_y','DDown_x','DDown_y']}
        y = {j: -chdata.loc[subset[i], j]/10 for j in ['DDown_x','DDown_y']}
        y['Angle'] = chdata.loc[subset[i], 'Angle']
#        x = {j: angle_t[subset[i]] for j in ['DUp_x','DUp_y','DDown_x','DDown_y','Angle']}
        x = {j: angle_t[subset[i]] for j in ['DDown_x','DDown_y','Angle']}
#        line_specs = {j: copy.deepcopy(plot_specs[i]) for j in ['DUp_x','DUp_y','DDown_x','DDown_y','Angle']}
        line_specs = {j: copy.deepcopy(plot_specs[i]) for j in ['DDown_x','DDown_y','Angle']}
#        line_specs['DUp_x']['color'] = '#03001e'
#        line_specs['DUp_y']['color'] = '#7303c0'
        line_specs['DDown_x']['color'] = '#ec38bc'
        line_specs['DDown_y']['color'] = '#EF3B36'
        line_specs['Angle']['color'] = '#4286f4'
        ax = plot_overlay_2d(ax, x, y, line_specs = line_specs)
        
    ax.set_xticks(np.linspace(0,0.14,8))
    ax.set_yticks(range(0,36,4))
    ax.axhline(1,color='k',linewidth=1)
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(grp[0])
    plt.show()
#%% 2D PLOTS        
#%% plot seat trajectories in B11 and B12 installations
models = ['Evenflo Embrace  CRABI','SNUGRIDE CRABI','Cybex Aton Q']
installs = ['B11','B12']
grouped = table.table.query_list('MODEL', models).table.query_list('INSTALL',installs).groupby(['SLED','MODEL'])
line_specs = {'Type 2 belt': {'linestyle': '-'},
              'UAS': {'linestyle': '--'}}

for grp in grouped:
    for i in line_specs:
        line_specs[i]['color'] = plot_specs[grp[0][0]]['color']
        
    x = arrange_by_group(grp[1], -chdata['Up_x'], 'INSTALL')
    y = arrange_by_group(grp[1], chdata['Up_y'], 'INSTALL')
#    x2 = arrange_by_group(grp[1], -chdata['DUp_x'], 'INSTALL')
#    y2 = arrange_by_group(grp[1], chdata['DUp_y'], 'INSTALL')

    
#    for z in x, y, x2, y2:
    for z in x, y:
        z['Type 2 belt'] = z.pop('B11')
        z['UAS'] = z.pop('B12')

    if x=={} or y=={}: continue
    fig, ax = plt.subplots()
#    ax = plot_overlay_2d(ax, x, y, line_specs=line_specs)
    ax = plot_overlay_2d(ax, x, y, line_specs=line_specs)
    ax = set_labels(ax, {'title': rename(grp[0][1]) + '\n(' + rename(grp[0][0]) + ')', 'xlabel': 'Excursion [mm]', 'ylabel': 'V. Displacement [mm]', 'legend': {}})
#    ax.set_xlim([0, 220])
#    ax.set_ylim([-220,0])
    ax = adjust_font_sizes(ax,{'ticklabels':20, 'title':24, 'axlabels':20, 'legend':20})
    ax.title.set_position([0.5, 1.05])
    plt.show()
    plt.close(fig)
#%% compare seat trajectories vs angle across benches/sleds
grouped = (table.query('DUMMY==\'Y2\'')
                .drop('SE16-0364')
                .table.query_list('MODEL',models)
                .table.query_list('INSTALL',['B11','B12'])
                .table.query_list('SLED',['old_accel'])
                .groupby(['MODEL']))
for grp in grouped:
    dx = arrange_by_group(grp[1],chdata['Angle'],'INSTALL')
    dy = arrange_by_group(grp[1],-chdata['DUp_x'],'INSTALL')
    ch_len = np.min(np.concatenate([dy[k].apply(len).values for k in dy]))
    mean_dx = pd.DataFrame({k: np.mean(np.vstack(dy[k].apply(lambda x: x[:ch_len]).values), axis=0) for k in dy})
    hline = ((mean_dx['B11']-mean_dx['B12']).abs()>=3).idxmax()
    if dx=={} or dy=={}: continue
    fig, ax = plt.subplots()
    ax = plot_overlay_2d(ax,dx,dy,line_specs=plot_specs)
    ax.axhline(mean_dx['B11'][hline])
    ax.axhline(mean_dx['B12'][hline])
    ax = set_labels(ax, {'title': grp[0], 'xlabel': 'Angle Change', 'ylabel': 'Excursion', 'legend': {}})
    plt.show()
    plt.close(fig)

#%%
grouped = (table.query('DUMMY==\'Y2\'')
                .drop('SE16-0364')
                .table.query_list('MODEL',models)
                .table.query_list('INSTALL',['B11','B12'])
                .table.query_list('SLED',['old_accel'])
                .groupby(['MODEL']))
for grp in grouped:
    dx = arrange_by_group(grp[1], chdata['DDown_x'], 'INSTALL')
    dy = arrange_by_group(grp[1], chdata['DUp_y'], 'INSTALL')
    ch_len = np.min(np.concatenate([dy[k].apply(len).values for k in dy]))
    mean_dy = pd.DataFrame({k: np.mean(np.vstack(dy[k].apply(lambda x: x[:ch_len]).values), axis=0) for k in dy})
    mean_dx = pd.DataFrame({k: np.mean(np.vstack(dx[k].apply(lambda x: x[:ch_len]).values), axis=0) for k in dx})
    plt.plot(mean_dx['B12']-mean_dx['B11'], label='excursion')
    plt.plot(mean_dy['B12']-mean_dy['B11'], label='v displ.')
    plt.axhline(0, color='k', linewidth=1)
    plt.title(grp[0])
    plt.legend()
    plt.show()
#%% REGRESSION
#%% lasso lars
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.preprocessing import StandardScaler

subset = table.table.query_list('INSTALL',['B11','B12']).query('SLED==\'old_accel\'')

drop = ['Min_12SEBE0000B6FO0D',
        'Max_Up_x',
        'DDown_x_at_DDown_y',
        'Tmin_12SEBE0000B6FO0D',
        'Tmin_S0SLED000000ACXD',
        'DDown_x_at_Angle',
        'energy_trapz',
        'energy_simpson',
        'DUp_x_at_DDown_y',
        'Head_3ms',
        'Min_DUp_x',
        'Max_12HEAD0000Y2ACRA',
        'DUp_x_at_Angle',
        'Min_12HEAD0000Y2ACXA',
        'Max_12HEAD0000Y2ACYA',
        'Min_12HEAD0000Y2ACYA',
        'Min_12CHST0000Y2ACRC',
        'Max_12CHST0000Y2ACYC',
        'Tmin_12CHST0000Y2ACRC',
        'Tmin_12HEAD0000Y2ACXA',
        'Tmax_12SEBE0000B6FO0D']
x = features.loc[subset.index].drop('Min_DDown_x', axis=1)
x = x.drop([i for i in drop if i in x.columns], axis=1)
x = x.dropna(axis=1, how='all')
y = features.loc[subset.index, 'Min_DDown_x']
for col in x:
    x[col] = x[col].replace(np.nan, x[col].mean())
keep_idx = y[~y.isna()].index
x = x.loc[keep_idx]
y = y.loc[keep_idx]

ss = StandardScaler()
x = pd.DataFrame(ss.fit_transform(x), index=x.index, columns=x.columns)
model = LassoLarsCV()
model = model.fit(x, y)
coefs = pd.Series(model.coef_, index=x.columns)
keep_cols = coefs[coefs.abs()>0]
print(keep_cols)
#%% regression
ch0_list = ['Chest_3ms']
plot_channels = ['Min_DDown_x',
                 'TDDown_y-Angle']

subset = (table.query('DUMMY==\'Y2\'')
#               .drop(['SE16-0253','SE16-0257','SE16-0351', 'SE16-0364'])
#               .table.query_list('MODEL',['PRONTO HIII-6-YR'])
               .table.query_list('INSTALL',['B11', 'B12'])
               .table.query_list('SLED',['new_accel', 'new_decel']))

#ch0_list = [i for i in features.columns if 'SEBE' not in i]
#plot_channels = ['Min_12CHST0000Y7DSXB']
#subset = table.query('DUMMY==\'Y7\' and CHEST_FIRST==False')
##               .table.query_list('INSTALL',['HB','LB'])
##               .table.query_list('SLED',['new_accel','new_decel']))
    
for ch0 in ch0_list:
    for ch in plot_channels:
        if ch0==ch: continue
        
        x = arrange_by_group(subset, features[ch], 'INSTALL')
        y = arrange_by_group(subset, features[ch0], 'INSTALL')

        match_groups(x,y)
        
        xname = rename(ch)
        yname = rename(ch0)
        xunit = re.search('\[.+\]',get_units(ch)).group()
        yunit = re.search('\[.+\]',get_units(ch0)).group()
        if x=={} or y=={}: continue
        if ch in ['Min_DDown_x']:
            for k in x:
                x[k] = x.pop(k).abs()
        elif ch0 in ['Min_DDown_x']:
            for k in y:
                y[k] = y.pop(k).abs()
        if ch in ['TDDown_y-Angle']:
            xunit = '[ms]'
        elif ch0 in ['TDDown_y-Angle']:
            yunit = '[ms]'
        
        rsq = {k: float(corr(x[k], y[k])) for k in x}
#        if max(rsq.values()) < 0.3 and min(rsq.values())>-0.3: continue
        renamed = {k: rename(k).replace('\n','') + ' R=' + str(rsq[k])[:5] for k in x}
        combined_rsq = corr(pd.concat([x[k] for k in x]),pd.concat([y[k] for k in y]))
        combined_rsq = str(combined_rsq)[:6] if combined_rsq<0 else str(combined_rsq)[:5]
        
        fig, ax = plt.subplots()
        ax = plot_scatter(ax, x, y, marker_specs={k: {'markersize': 10} for k in x})
        ax = set_labels(ax, {'title': 'Combined R=' + combined_rsq,'xlabel': ' '.join((xname,xunit)), 'ylabel': ' '.join((yname, yunit)),'legend': {'bbox_to_anchor': (0.95,-0.36)}})
#        ax = adjust_font_sizes(ax,{'ticklabels':20,'axlabels':20,'legend':20,'title':24})
        ax = adjust_font_sizes(ax,{'ticklabels':22,'axlabels':24,'legend':24,'title':28})
        ax.set_ylim([40, 60])
        rename_legend(ax, renamed)
        plt.show()
        
#        plotly_fig = plot_scatter_with_labels(x,y)
#        plot(plotly_fig)
#        plt.close(fig)

#%%
preds = pd.read_csv(directory + 'predictions.csv',index_col=0)
x = {'pred': pd.Series(range(len(preds)),index=preds.index),
     'act': pd.Series(range(len(preds)),index=preds.index)}
y = {'pred': preds['pred'],
     'act': preds['act']}
plotly_fig = plot_scatter_with_labels(x,y)
plot(plotly_fig)

#%% plot faro
faro_files = glob.glob(directory + 'Faro//*.csv')
faro_points = {}
trace = []
for file in faro_files:
    file_points = pd.read_csv(file)
    se = re.search('SE\d{2}-\d{4}_{0,1}\d{0,1}',file).group()
    faro_points[se] = file_points
    t = go.Scatter3d(x=file_points['x'],
                     y=file_points['y'],
                     z=file_points['z'],
                     text=file_points['point_definition_id'],
                     name=se,
                     mode='markers')
    trace.append(t)
    
layout = go.Layout(scene=dict(xaxis=axis_template,
                              yaxis=axis_template,
                              zaxis=axis_template))
fig = go.Figure(data=trace, layout=layout)
plot(fig)
