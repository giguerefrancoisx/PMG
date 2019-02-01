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
import lightgbm as lgb
import glob

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
#%%
features = get_all_features(csv_write=False,json_write=False)

#%%
lgb_drop = ['Min_S0SLED000000ACXD',
            'Tmin_S0SLED000000ACXD',
            'Max_12HEAD0000Y7ACXA',
            'Tmax_12HEAD0000Y7ACXA',
            'Min_12HEAD0000Y7ACZA',
            'Tmin_12HEAD0000Y7ACZA',
            'Min_12HEAD0000Y7ACRA',
            'Tmin_12HEAD0000Y7ACRA',
            'Max_12NECKUP00Y7FOXA',
            'Tmax_12NECKUP00Y7FOXA',
            'Min_12NECKUP00Y7FORA',
            'Tmin_12NECKUP00Y7FORA',
            'Max_12CHST0000Y7ACXC',
            'Tmax_12CHST0000Y7ACXC',
            'Max_12CHST0000Y7ACZC',
            'Tmax_12CHST0000Y7ACZC',
            'Min_12CHST0000Y7ACRC',
            'Tmin_12CHST0000Y7ACRC',
            'Max_12CHST0000Y7DSXB',
            'Tmax_12CHST0000Y7DSXB',
            'Max_12LUSP0000Y7FOXA',
            'Tmax_12LUSP0000Y7FOXA',
            'Max_12PELV0000Y7ACXA',
            'Tmax_12PELV0000Y7ACXA',
            'Max_12PELV0000Y7ACZA',
            'Tmax_12PELV0000Y7ACZA',
            'Min_12PELV0000Y7ACRA',
            'Tmin_12PELV0000Y7ACRA',
            'Min_12SEBE0000B3FO0D',
            'Tmin_12SEBE0000B3FO0D',
            'Min_12SEBE0000B6FO0D',
            'Tmin_12SEBE0000B6FO0D',
            'X12CHST0000Y7ACXC_at_12CHST0000Y7ACRC',
            'X12HEAD0000Y7ACZA_at_12HEAD0000Y7ACRA',
            'Min_12LUSP0000Y7FOXA']

lgb_drop2 = ['Head_Excursion',
             'Knee_Excursion',
             'Head_3ms',
             'Chest_3ms',
             'Min_12CHST0000Y7DSXB_trunc']

lgb_drop = lgb_drop + lgb_drop2

x = (features.loc[table.query('DUMMY==\'Y7\'').table.query_list('INSTALL',['HB','LB']).index]
             .drop(lgb_drop,axis=1)
             .dropna(axis=1,how='all'))
y = x.pop('Min_12CHST0000Y7DSXB')

train_data = lgb.Dataset(x, label=y)

param = {'objective': 'regression',
         'bagging_fraction': 0.8}

n_rounds = 10
importance = pd.DataFrame(index=range(n_rounds), columns=x.columns)

for i in range(n_rounds):
    model = lgb.train(param, train_data)
    importance.loc[i] = model.feature_importance()
#    lgb.plot_importance(model, figsize=(10,8))
print(importance.mean().sort_values())
#%% get tmin_chest - tmin_pelvis
features['Tmin_12CHST0000Y7ACXC-12PELV0000Y7ACXA'] = features['Tmin_12CHST0000Y7ACXC']-features['Tmin_12PELV0000Y7ACXA']
se_neg = features['Tmin_12CHST0000Y7ACXC-12PELV0000Y7ACXA']<0
table['CHEST_FIRST'] = se_neg
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
    diff[c].to_csv(directory + c + '.csv')

#%% CATEGORICAL
#%% bar plots comparing different responses on the two benches or on the two sleds
plot_channels = ['Max_12SEBE0000B3FO0D',
                 'Max_12NECKUP00Y7FOZA',
                 'Min_12CHST0000Y7DSXB',
                 'Max_12PELV0000Y7ACRA',
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
                 'Max_12HEAD0000Y2ACRA',
                 'Max_12CHST0000Y2ACRC',
                 'X12CHST0000Y2ACXC_at_12CHST0000Y2ACRC',
                 'X12CHST0000Y2ACZC_at_12CHST0000Y2ACRC',
                 'X12HEAD0000Y2ACXA_at_12HEAD0000Y2ACRA',
                 'X12HEAD0000Y2ACZA_at_12HEAD0000Y2ACRA',
                 'Tmin_DDown_y',
                 'Tmax_Angle']

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
            add_stars(ax, ax.get_xticks(), p[install_order[d]], y, fontsize=24)
            
            # set labels and font sizes
            ax = set_labels(ax, {'title': rename(ch), 'ylabel': get_units(ch)})
            ax = adjust_font_sizes(ax, {'ticklabels':20,'title':24,'ylabel':20})
#            ax.legend(ncol=2,bbox_to_anchor=(1,1))
#            ax = adjust_font_sizes(ax, {'ticklabels':16,'title':20,'legend':18,'ylabel':18})
            ax.set_ylim([0, 1.2*max(y)])
            ax.set_xticklabels([rename(x) for x in installs])
            plt.show()
            plt.close(fig)

#%% bar plots model by model
plot_channels = ['X12CHST0000Y2ACXC_at_12CHST0000Y2ACRC',
                 'X12CHST0000Y2ACZC_at_12CHST0000Y2ACRC',
                 'TDDown_y-Angle']
subset = (table.query('INSTALL==\'C1\'')
                .table.query_list('SLED',['new_accel','new_decel']))

for ch in plot_channels:
    x = arrange_by_group(subset, features[ch], 'SLED','MODEL')
    fig, ax = plt.subplots()
    ax = plot_bar(ax, x, plot_specs=plot_specs)
    ax = set_labels(ax, {'title': rename(ch), 'ylabel': get_units(ch)})
    ax = adjust_font_sizes(ax, {'ticklabels': 16, 'title': 20, 'axlabels': 18})
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
models = ['Evenflo Embrace  CRABI','SNUGRIDE CRABI','Cybex Aton Q']
installs = ['B11','B12']
cols = ['DUp_x','DUp_y','DDown_x','DDown_y']
grouped = table.table.query_list('MODEL',models).table.query_list('INSTALL',installs).groupby(['MODEL','SLED'])
distances = pd.Series(index=grouped.groups.keys())

# get distances
for grp in grouped:
    subgroup = grp[1].groupby('INSTALL').groups
    distances[grp[0]] = get_distance(chdata.loc[subgroup['B11'], cols], chdata.loc[subgroup['B12'], cols])  
#    distances[grp[0]] = get_distance(chdata.loc[subgroup['B12'],cols], chdata.loc[subgroup['B11'], cols])  
distances = distances.unstack().to_dict()

for c in comparison:
    x = {k: pd.DataFrame(distances[k],index=[0]) for k in ['new_accel', c]}   
    colours = {'new_accel': plot_specs['new_accel']['color'],
               c: plot_specs[c]['color']}
    fig, ax = plt.subplots()
    ax = plot_bar(ax, x, errorbar=False, plot_specs=plot_specs)
    ax = set_labels(ax, {'ylabel': 'Distance Metric', 'title': '(TBD)','legend': {}})
#    ax = adjust_font_sizes(ax,{'ticklabels': 18,'title': 20,'legend':18,'ylabel':18})

#%% get euclidean distance between channels (not RFCRS displacement)
plot_channels = ['12LUSP0000Y7MOZA']
grouped = (table.query('DUMMY==\'Y7\'') 
                .groupby(['INSTALL','MODEL']))

for ch in plot_channels:
    for c in comparison:
        dist = pd.DataFrame()
        for grp in grouped: 
            subgroup = grp[1].groupby('SLED').groups
            x = chdata.loc[subgroup['new_accel'], ch]
            if c in subgroup:
                y = chdata.loc[subgroup[c], ch]
            else: 
                continue
            dist.at[grp[0][1], grp[0][0]] = np.sqrt((unpack(x).mean(axis=1)-unpack(y).mean(axis=1))**2).sum()
        fig, ax = plt.subplots()
        plot_bar(ax, {'label': dist}, errorbar=True, order=install_order['Y7'], plot_specs = {'label': {'color': (0.65,0.65,0.65)}})
        ax = set_labels(ax, {'title': 'Comparison with ' + c, 'ylabel': 'Distance Metric'})
        ax = adjust_font_sizes(ax,{'ticklabels':18,'title':20,'ylabel':18})

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
plot_channels = ['12CHST0000Y7ACYC',
                '12LUSP0000Y7FOYA',
                '12LUSP0000Y7MOXA',
                '12LUSP0000Y7MOZA']
grouped = (table.table.query_list('INSTALL',['HB','LB'])
                .table.query_list('SLED',['new_accel', 'new_decel'])
#                .table.query_list('MODEL',['PRONTO HIII-6-YR'])
                .groupby(['MODEL', 'INSTALL']))
for ch in plot_channels:
    for subset in grouped:
        x = arrange_by_group(subset[1], chdata[ch], 'SLED')
        if len(x)==0: continue
        fig, ax = plt.subplots()
        ax = plot_overlay(ax, t, x, line_specs=plot_specs)
#        ax.plot(t, chdata.at['SE16-0337',ch], color='#00FFFF', linewidth=1.5, linestyle='--', label='Pronto high')
#        ax.plot(t, chdata.at['SE16-0338',ch], color='#FF00FF', linewidth=1, linestyle='--', label='Pronto low')
        ax = set_labels(ax, {'title': ch + '\n' + str(subset[0]), 'legend': {'bbox_to_anchor': (1,1)}})
        plt.show()
        plt.close(fig)     

#%% plot multiple channels on the same axis
plot_channels = ['12CHST0000Y2ACXC',
                 '12CHST0000Y2ACZC',
                 '12CHST0000Y2ACRC']
grouped = (table.query('DUMMY==\'Y2\'')
                .table.query_list('SLED',['old_accel','new_accel'])
#                .table.query_list('MODEL',models)
                .table.query_list('INSTALL',['B11','B12'])
                .groupby(['INSTALL','SLED','MODEL']))
for grp in grouped:
    fig, ax = plt.subplots()
    for ch in plot_channels:
#        x = {ch[14] if ch[14] in ['X','Y','Z'] else 'Resultant': chdata.loc[grp[1].index, ch]}
        x = {ch: chdata.loc[grp[1].index, ch]}
        if len(x)==0: continue
        ax = plot_overlay(ax,t,x)
    ax.axhline(0, color='k', linewidth=1)
#    ax.set_yticks(range(-40, 80, 20))
#    ax = set_labels(ax, {'title': rename(grp[0][0]).replace('\n','') + ', ' + rename(grp[0][2]), 'legend': {}, 'xlabel': 'Time [s]', 'ylabel': 'Acceleration [g]'})
    ax = set_labels(ax, {'title': grp[0], 'legend': {'bbox_to_anchor': (1,1)}, 'xlabel': 'Time [s]', 'ylabel': 'Acceleration [g]'})
    ax = adjust_font_sizes(ax,{'ticklabels':18, 'title':20, 'axlabels':18, 'legend':18})
    plt.show()
    plt.close(fig)
#%% plot overlays of excursions vs. time comparing installations
plot_channels = ['DDown_y',
                 'DDown_x']
models = ['Evenflo Embrace  CRABI','SNUGRIDE CRABI','Cybex Aton Q']
installs = ['B0']
grouped = (table.query('DUMMY==\'Y2\'')
#                .table.query_list('MODEL',models)
                .table.query_list('INSTALL',installs)
                .groupby(['MODEL']))
line_specs = {'B11': {'linestyle': '-'},
              'B12': {'linestyle': '--'}}

for ch in plot_channels:
    for grp in grouped:
#        for i in line_specs:
#            line_specs[i]['color'] = plot_specs[grp[0][0]]['color']
        # x is time; y is ch 
        x = arrange_by_group(grp[1], angle_t, 'SLED')
        y = arrange_by_group(grp[1], chdata[ch], 'SLED')
        if x=={} or y=={}: continue 
        fig, ax = plt.subplots()
        ax = plot_overlay_2d(ax, x, y, line_specs=line_specs)
        ax = set_labels(ax, {'title': str(grp[0]) + ' ' + ch, 'xlabel': 'Time', 'ylabel': 'Displacement', 'legend': {}})
        plt.show()
        plt.close(fig)

#%% plot time series one test at a time
subset = (table.query('DUMMY==\'Y2\'')
               .table.query_list('INSTALL',['C1'])
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
        y = {j: -chdata.loc[subset[i], j]/10 for j in ['DUp_x','DUp_y','DDown_x','DDown_y']}
        y['Angle'] = chdata.loc[subset[i], 'Angle']
        x = {j: angle_t[subset[i]] for j in ['DUp_x','DUp_y','DDown_x','DDown_y','Angle']}
        line_specs = {j: copy.deepcopy(plot_specs[i]) for j in ['DUp_x','DUp_y','DDown_x','DDown_y','Angle']}
        line_specs['DUp_x']['color'] = '#03001e'
        line_specs['DUp_y']['color'] = '#7303c0'
        line_specs['DDown_x']['color'] = '#ec38bc'
        line_specs['DDown_y']['color'] = '#EF3B36'
        line_specs['Angle']['color'] = '#4286f4'
        ax = plot_overlay_2d(ax, x, y, line_specs = line_specs)
        
    ax.set_xticks(np.linspace(0,0.14,8))
    ax.set_yticks(range(0,24,2))
    ax.axhline(1,color='k',linewidth=1)
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(grp[0])
    plt.show()
#%% 2D PLOTS
#%% plot two channels in 2D
plot_channels = ['12LUSP0000Y7MOXA',
                 '12LUSP0000Y7MOZA']

grouped = table.groupby('SLED')
for c in comparison:
    subset = pd.concat((grouped.get_group('new_accel'), grouped.get_group(c))).groupby(['DUMMY','INSTALL','MODEL'])
    for grp in subset:
        x = arrange_by_group(grp[1],chdata[plot_channels[0]],'SLED')
        y = arrange_by_group(grp[1],chdata[plot_channels[1]],'SLED')
        if x=={} or y=={}: continue
        fig, ax = plt.subplots()
        ax = plot_overlay_2d(ax, x, y, line_specs=plot_specs)
        ax.axvline(0,color='k',linewidth=1)
        ax.axhline(0,color='k',linewidth=1)
        ax = set_labels({'title': grp[0], 'xlabel': plot_channels[0], 'ylabel': plot_channels[1], 'legend': {'bbox_to_anchor': (1,1)}})
        plt.show()
        plt.close(fig)
        
#%% plot seat trajectories in B11 and B12 installations
models = ['Evenflo Embrace  CRABI','SNUGRIDE CRABI','Cybex Aton Q']
installs = ['B11','B12']
grouped = table.table.query_list('MODEL', models).table.query_list('INSTALL',installs).groupby(['SLED','MODEL'])
line_specs = {'Type 2 belt': {'linestyle': '-'},
              'UAS': {'linestyle': '--'}}

for grp in grouped:
    for i in line_specs:
        line_specs[i]['color'] = plot_specs[grp[0][0]]['color']
        
    x = arrange_by_group(grp[1], -chdata['DDown_x'], 'INSTALL')
    y = arrange_by_group(grp[1], chdata['DDown_y'], 'INSTALL')
    x2 = arrange_by_group(grp[1], -chdata['DUp_x'], 'INSTALL')
    y2 = arrange_by_group(grp[1], chdata['DUp_y'], 'INSTALL')

    
    for z in x, y, x2, y2:
        z['Type 2 belt'] = z.pop('B11')
        z['UAS'] = z.pop('B12')

    if x=={} or y=={}: continue
    fig, ax = plt.subplots()
    ax = plot_overlay_2d(ax, x, y, line_specs=line_specs)
    ax = plot_overlay_2d(ax, x2, y2, line_specs=line_specs)
    ax = set_labels(ax, {'title': rename(grp[0][1]) + ' (' + rename(grp[0][0]) + ')', 'xlabel': 'Excursion [mm]', 'ylabel': 'V. Displacement [mm]', 'legend': {}})
    ax.set_xlim([0, 220])
    ax.set_ylim([-220,0])
    ax = adjust_font_sizes(ax,{'ticklabels':18, 'title':20, 'axlabels':18, 'legend':18})
    plt.show()
    plt.close(fig)
#%% compare seat trajectories vs angle across benches/sleds
grouped = (table.query('DUMMY==\'Y2\'')
                .drop('SE16-0364')
                .table.query_list('INSTALL',['B0','C1'])
                .table.query_list('SLED',['new_accel','new_decel'])
                .groupby(['MODEL','INSTALL']))
for grp in grouped:
    dx = arrange_by_group(grp[1],chdata['Angle'],'SLED')
    dy = arrange_by_group(grp[1],-chdata['DDown_x'],'SLED')
    if dx=={} or dy=={}: continue
    fig, ax = plt.subplots()
    ax = plot_overlay_2d(ax,dx,dy,line_specs=plot_specs)
    ax = set_labels(ax, {'title': grp[0], 'xlabel': 'Angle Change', 'ylabel': 'Excursion', 'legend': {}})
    plt.show()
    plt.close(fig)

#%% REGRESSION
#%% regression
ch0_list = ['Max_12LUSP0000Y7MOZA']
plot_channels = ['Max_12NECKUP00Y7FOZA']

subset = (table.query('DUMMY==\'Y7\'')
#               .drop(['SE16-0253','SE16-0257','SE16-0351', 'SE16-0364'])
#               .table.query_list('MODEL',['PRONTO HIII-6-YR'])
               .table.query_list('INSTALL',['HB','LB']))
#               .table.query_list('SLED',['new_accel','new_decel','old_accel']))

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

        if x=={} or y=={}: continue
        match_groups(x,y)
        
        rsq = {k: float(corr(x[k], y[k])) for k in x}
#        if max(rsq.values()) < 0.3 and min(rsq.values())>-0.3: continue
        renamed = {k: rename(k).replace('\n','') + ' R=' + str(rsq[k])[:5] for k in x}
        combined_rsq = corr(pd.concat([x[k] for k in x]),pd.concat([y[k] for k in y]))
        combined_rsq = str(combined_rsq)[:6] if combined_rsq<0 else str(combined_rsq)[:5]
        
        fig, ax = plt.subplots()
        ax = plot_scatter(ax, x, y)#, marker_specs=plot_specs)
        ax = set_labels(ax, {'title': 'Combined R=' + combined_rsq,'xlabel': rename(ch), 'ylabel': rename(ch0),'legend': {'bbox_to_anchor': (1,1)}})
        ax = adjust_font_sizes(ax,{'ticklabels':18,'axlabels':18,'legend':16,'title':20})
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
