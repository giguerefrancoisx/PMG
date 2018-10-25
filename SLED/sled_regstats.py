# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:44:50 2018

@author: tangk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PMG.COM.plotfuns import *
from PMG.COM.helper import r2, is_all_nan, query_list
from PMG.COM.get_props import peakval, get_ipeak, get_angle, get_shifted, smooth_data
from PMG.read_data import initialize, get_se_angle
from PMG.COM.arrange import sep_by_peak, align
import seaborn
import json
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

plotfigs = 1
savefigs = 0
writefiles = 0
writejson = 0
cutoff = range(100,1600)

directory = 'P:\\SLED\\'

channels = ['12HEAD0000Y2ACXA',
            '12HEAD0000Y2ACZA',
            '12HEAD0000Y2ACRA',
            '12CHST0000Y2ACXC',
            '12CHST0000Y2ACZC',
            '12CHST0000Y2ACRC',
            '12PELV0000Y2ACXA',
            '12PELV0000Y2ACZA',
            '12PELV0000Y2ACRA',
            '12HEAD0000Y7ACXA',
            '12HEAD0000Y7ACZA',
            '12HEAD0000Y7ACRA',
            '12NECKUP00Y7FOXA',
            '12NECKUP00Y7FOZA',
            '12NECKUP00Y7FORA',
            '12NECKUP00Y7MOXB',
            '12CHST0000Y7ACXC',
            '12CHST0000Y7ACZC',
            '12CHST0000Y7ACRC',
            '12CHST0000Y7DSXB',
            '12LUSP0000Y7FOXA',
            '12LUSP0000Y7FOZA',
            '12LUSP0000Y7MOXA',
            '12LUSP0000Y7MOYA',
            '12PELV0000Y7ACXA',
            '12PELV0000Y7ACZA',
            '12PELV0000Y7ACRA',
            '12SEBE0000B3FO0D',
            '12SEBE0000B6FO0D']

exclude = ['SE16-1012_2',
           'SE16-1014_2',
           'SE17-1015_2',
           'SE17-1016_2',
           'SE16-0399',
           'SE16-0402',
           'SE16-0395',
           'SE16-0403',
           'SE17-1025_2',
           'SE17-1026_2']

responses = [['Head 3ms','Chest 3ms'],
             ['Head Excursion','Knee Excursion']]
comparison = ['old_accel','new_decel']

width = 0.35
install_name = {'H1':'FFCRS installed with Type 2 Belt',
                'H3':'FFCRS installed with UAS',
                'HB':'High-Back Booster',
                'LB':'Low-Back Booster',
                'B0':'RFCRS with no Base',
                'B11':'RFCRS installed with \nbase and Type 2 belt',
                'B12':'RFCRS installed with \nbase and UAS',
                'C1': 'Convertible RFCRS \ninstalled with Type 2 belt'}
dummies = ['Y7','Y2']

table, t, chdata = initialize(directory, channels,cutoff,drop=exclude,query='(DUMMY==\'Y2\' or DUMMY==\'Y7\') and DATA==\'YES\'')
se_angles = get_se_angle(chdata.index).apply(get_angle,axis=1).rename('Angle')
angle_t = get_se_angle(chdata.index)['Time']/1000

displacement = get_se_angle(chdata.index).applymap(lambda x: x-x[0])[['Up_x', 'Up_y', 'Down_x', 'Down_y']].rename(lambda x: 'D'+x,axis=1)
#x = pd.concat([angle_t,se_angles],axis=1).apply(lambda x: align(t,x['Time'],x['Angle']),axis=1)
chdata = chdata.join(se_angles.apply(lambda x: x-x[0]),sort=True)
chdata = pd.concat([chdata,displacement],axis=1).applymap(lambda x: np.array([x]) if type(x)==np.float else x)
chdata = chdata.join(chdata[['12CHST0000Y7ACXC','12PELV0000Y7ACXA']].apply(lambda x: x['12CHST0000Y7ACXC']-x['12PELV0000Y7ACXA'],axis=1).rename('Chest-Pelvis_tvar'))
chdata = chdata.join(chdata[['12HEAD0000Y7ACXA','12PELV0000Y7ACXA']].apply(lambda x: x['12HEAD0000Y7ACXA']-x['12PELV0000Y7ACXA'],axis=1).rename('Head-Pelvis_tvar'))
#chdata['12PELV0000Y7ACXA'] = chdata['12PELV0000Y7ACXA'].apply(smooth_data)
#chdata['12PELV0000Y7ACZA'] = chdata['12PELV0000Y7ACZA'].apply(smooth_data)
chdata.at['SE16-0414','12PELV0000Y7ACRA'] = [np.nan]
#%%

#%% get peaks, etc    
def get_stattable(csv_write=False,json_write=False):                    
    def i_to_t(i):
        if not np.isnan(i):
            return t[int(i)]
        else:
            return np.nan
        
    mins, maxs = sep_by_peak(chdata.applymap(peakval))
    
    tmins,tmaxs = sep_by_peak(chdata.applymap(get_ipeak))
    mins = mins.rename(lambda name:('Min_' + name),axis=1)
    maxs = maxs.rename(lambda name:('Max_' + name),axis=1)
    tmins = tmins.rename(lambda name:('Tmin_' + name),axis=1).applymap(i_to_t)
    tmaxs = tmaxs.rename(lambda name:('Tmax_' + name),axis=1).applymap(i_to_t)

    stattable = pd.concat((mins,
                           maxs,
                           tmins,
                           tmaxs),axis=1)
    stattable = stattable.join(table[['Head 3ms','Chest 3ms','Head Excursion','Knee Excursion']].rename(lambda x: x.replace(' ','_'),axis=1),how='right').rename_axis('SE')
    stattable = stattable.join(pd.DataFrame(mins['Min_12CHST0000Y7ACXC']-mins['Min_12PELV0000Y7ACXA'],index=stattable.index,columns=['Chest-Pelvis']))
    
    stattable.to_csv(directory + 'stattable.csv')
    
    to_JSON = {'project_name': 'FMVSS213_sled_comparison',
               'directory'   : directory}
    
    if writejson:
        with open(directory+'params.json','w') as json_file:
            json.dump(to_JSON,json_file)
    
    return stattable

stattable = get_stattable(csv_write=False,json_write=False)
    
#%%
with open(directory+'params.json','r') as json_file:
    to_JSON = json.load(json_file)

#pd.DataFrame(to_JSON['res']['Y7']['bench']['p']).set_index('_row',drop=True)

#%% comparison of 3ms clips and excursions
indices = np.arange(len(responses))
for d in dummies:
    if d=='Y7':
        installs = ['H1','H3','HB','LB']
    else:
        installs = ['B0','B11','B12','C1']
    for c in comparison:
        for r in responses:
            if r==['Head Excursion','Knee Excursion'] and d=='Y2':
                continue
            for i in installs:
                
                new_accel = table.query('INSTALL==\'' + i + '\' & SLED==\'new_accel\'')[r]
                old_accel = table.query('INSTALL==\'' + i + '\' & SLED==\''+c+'\'')[r]
                
                if len(new_accel)==0 or len(old_accel)==0:
                    continue
                
                if c=='old_accel':
                    p = pd.DataFrame(to_JSON['res'][d]['bench']['p']).set_index('_row',drop=True)
#                    variance = pd.DataFrame(to_JSON['res'][d]['bench']['var']).set_index('_row',drop=True)
                else:
                    p = pd.DataFrame(to_JSON['res'][d]['sled']['p']).set_index('_row',drop=True)   
#                    variance = pd.DataFrame(to_JSON['res'][d]['sled']['var']).set_index('_row',drop=True)   
                    
                p = p[[s.replace(' ','_') for s in r]].loc[i]
#                variance = np.sqrt(variance[[s.replace(' ','_') for s in r]].loc[i])
                
                fig, ax = plt.subplots()
                bar1 = ax.bar(indices-width/2,new_accel.mean(),width,yerr=new_accel.std(),color='IndianRed',label='New Bench',capsize=10,error_kw={'elinewidth':2,'capthick':2})
                if c=='old_accel':
                    bar2 = ax.bar(indices+width/2,old_accel.mean(),width,yerr=old_accel.std(),color='SkyBlue',label='Old Bench',capsize=10,error_kw={'elinewidth':2,'capthick':2})
                else:
                    bar2 = ax.bar(indices+width/2,old_accel.mean(),width,yerr=old_accel.std(),color='#589F58',label='Decel Sled',capsize=10,error_kw={'elinewidth':2,'capthick':2})
                
                for l, k in enumerate(p):
                    if k<0.05:
                        ax.text(indices[l],1.02*pd.concat((new_accel.max(),old_accel.max()),axis=1).max(axis=1).values[l],'*',fontsize=24)

                ax.set_xticks(indices)
                ax.set_xticklabels(r)
                ax.set_title(install_name[i])
                if r==['Head Excursion','Knee Excursion']:
                    ax.set_ylim([0, 850])
                elif r==['Head 3ms','Chest 3ms'] and d=='Y7':
                    ax.set_ylim([0, 80])
                elif r==['Head 3ms','Chest 3ms'] and d=='Y2':
                    ax.set_ylim([0,90])
                else:
                    ax.set_ylim([0,1.1*np.max(pd.concat((new_accel.max(),old_accel.max()),axis=1).max(axis=1).values)])
                
                ax.legend(bbox_to_anchor=(1,0.6))
                ax = adjust_font_sizes(ax, {'ticklabels': 18, 'title': 20, 'legend': 18})

#%% comparison of shoulder belt load
plot_channels = ['Max_12SEBE0000B3FO0D',
                 'Max_12NECKUP00Y7FORA',
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
                 'Min_DDown_y']
channel_names = {'Max_12SEBE0000B3FO0D': 'Peak Shoulder Belt Load',
                 'Max_12NECKUP00Y7FORA': 'Peak Neck Resultant Load',
                 'Min_12CHST0000Y7DSXB': 'Peak Chest Deflection',
                 'Head_Excursion': 'Head Excursion',
                 'Knee_Excursion': 'Knee Excursion',
                 'Head_3ms': 'Head 3ms Clip',
                 'Chest_3ms': 'Chest 3ms Clip',
                 'Max_12PELV0000Y7ACRA': 'Peak Pelvis Resultant Acceleration',
                 'Max_Angle': 'Angle Change',
                 'Min_DUp_x': 'Peak Excursion (Upper Target)',
                 'Min_DDown_x': 'Peak Excursion (Lower Target)',
                 'Min_DUp_y': 'Peak Vertical Displacement \n(Upper Target)',
                 'Min_DDown_y': 'Peak Vertical Displacement \n(Lower Target)'}
for d in dummies:
    for c in comparison:
        for ch in plot_channels:
            if d=='Y2' and 'Y7' in ch:
                continue
            elif d=='Y2' and 'Excursion' in ch:
                continue
            elif d=='Y7' and (('Y2' in ch) or ('Angle' in ch) or ('_x' in ch) or ('_y' in ch)):
                continue
            if d=='Y7' and 'SEBE' in ch:
                installs = ['H1','HB','LB']
            elif d=='Y7':
                installs = ['H1','H3','HB','LB']
            elif d=='Y2' and 'SEBE' in ch:
                installs = ['C1','B0','B11']
            elif d=='Y2':
                installs = ['C1','B0','B11','B12']
            if d=='Y7' and c=='new_decel':
                installs = ['HB','LB']
                
            
            if c=='old_accel':
                pvals = pd.DataFrame(to_JSON['res'][d]['bench']['p']).set_index('_row',drop=True)[ch]
            else:
                pvals = pd.DataFrame(to_JSON['res'][d]['sled']['p']).set_index('_row',drop=True)[ch]
            indices = np.arange(len(installs))
            
            means_new = []
            means_old = []
            stds_new = []
            stds_old = []
            for i in installs: 
                i_new = table.query('INSTALL==\'' + i + '\' & SLED==\'new_accel\'').index
                i_old = table.query('INSTALL==\'' + i + '\' & SLED==\'' + c + '\'').index
                
                new = stattable.loc[i_new,ch]
                old = stattable.loc[i_old,ch]
                
                new = new.dropna()
                old = old.dropna()
                
    #            if ch=='Max_12PELV0000Y7ACRA' and i=='HB':
    #                old = old.drop('SE16-0414')
                
                if len(new)==0 or len(old)==0:
                    means_new.append(np.nan)
                    means_old.append(np.nan)
                    stds_new.append(np.nan)
                    stds_old.append(np.nan)
                    continue
                
                means_new.append(new.mean())
                means_old.append(old.mean())
                stds_new.append(new.std())
                stds_old.append(old.std())
            
            fig, ax = plt.subplots()
            means_new = [abs(j) for j in means_new]
            means_old = [abs(j) for j in means_old]
            stds_new = [min(stds_new[i],means_new[i]) for i in range(len(stds_new))]
            stds_old = [min(stds_old[i],means_old[i]) for i in range(len(stds_old))]
            
            bar1 = ax.bar(indices-width/2,means_new,width,yerr=stds_new,color='IndianRed',label='New Bench',capsize=6,error_kw={'elinewidth':2,'capthick':2})
            if c=='old_accel':
                bar2 = ax.bar(indices+width/2,means_old,width,yerr=stds_old,color='SkyBlue',label='Old Bench',capsize=6,error_kw={'elinewidth':2,'capthick':2})
            else:
                bar2 = ax.bar(indices+width/2,means_old,width,yerr=stds_old,color='#589F58',label='Decel Sled',capsize=6,error_kw={'elinewidth':2,'capthick':2})

                
            for j in indices:
                if pvals[installs[j]] < 0.05:
                    y = max(means_old[j],means_new[j])+max(stds_old[j],stds_new[j])
                    ax.text(j,1.05*y,'*',fontsize=24)
            
            ax.set_xticks(indices)
            ax.set_xticklabels(installs)
            ax.set_title(channel_names[ch])
            if 'FO' in ch:
                ax.set_ylabel('Force [N]')
            elif 'AC' in ch:
                ax.set_ylabel('Acceleration [g]')
            elif 'Excursion' in ch:
                ax.set_ylabel('Excursion [mm]')
            elif '3ms' in ch:
                ax.set_ylabel('Acceleration [g]')
            elif 'DS' in ch:
                ax.set_ylabel('Deflection [mm]')
            elif 'Angle' in ch:
                ax.set_ylabel('Angle Change [deg]')
            elif '_x' in ch or '_y' in ch:
                ax.set_ylabel('Displacement [mm]')
            ax.legend(bbox_to_anchor=(1,0.6))
            ax = adjust_font_sizes(ax, {'ticklabels':18,'title':20,'legend':18,'ylabel':18})
            ax.set_ylim([0, 1.2*(max(means_old)+max(stds_old))])

#%%         
plot_channels = ['Max_12SEBE0000B3FO0D',
                 'Max_12NECKUP00Y7FORA',
                 'Min_12CHST0000Y7DSXB',
                 'Max_12PELV0000Y7ACRA',
                 'Head_Excursion',
                 'Knee_Excursion',
                 'Head_3ms',
                 'Chest_3ms']
channel_names = {'Max_12SEBE0000B3FO0D': 'Peak Shoulder Belt Load',
                 'Max_12NECKUP00Y7FORA': 'Peak Neck Resultant Load',
                 'Min_12CHST0000Y7DSXB': 'Peak Chest Deflection',
                 'Head_Excursion': 'Head Excursion',
                 'Knee_Excursion': 'Knee Excursion',
                 'Head_3ms': 'Head 3ms Clip',
                 'Chest_3ms': 'Chest 3ms Clip',
                 'Max_12PELV0000Y7ACRA': 'Peak Pelvis Resultant Acceleration'}

i_new = table.query('MODEL==\'PRONTO HIII-6-YR\' and SLED==\'new_accel\'').index[0]
i_old = table.query('MODEL==\'PRONTO HIII-6-YR\' and SLED==\'new_decel\'').index[0]

for ch in plot_channels:
    fig, ax = plt.subplots()
    ax.bar(['Accel sled'],stattable.loc[i_new,ch],color='IndianRed')
    ax.bar(['Decel sled'],stattable.loc[i_old,ch],color='#589F58')
    ax.set_title(channel_names[ch])
    if 'FO' in ch:
        ax.set_ylabel('Force [N]')
    elif 'AC' in ch:
        ax.set_ylabel('Acceleration [g]')
    elif 'Excursion' in ch:
        ax.set_ylabel('Excursion [mm]')
    elif '3ms' in ch:
        ax.set_ylabel('Acceleration [g]')
    elif 'DS' in ch:
        ax.set_ylabel('Deflection [mm]')
    ax = adjust_font_sizes(ax,{'ticklabels':18,'title':20,'ylabel':18})
    print(stattable.loc[i_old,ch]-stattable.loc[i_new,ch])
        
#%% plot overlays model-by-model
from PMG.COM.helper import is_all_nan

plot_channels = [
            '12NECKUP00Y7FOXA',
            '12NECKUP00Y7FOZA',
            '12LUSP0000Y7FOXA']
for d in dummies:
    if d=='Y7':
        installs = ['H1','H3','HB','LB']
    else:
        installs = ['B0','B11','B12','C1']
        
    for c in comparison:
        for ch in plot_channels:
            if d=='Y2' and 'Y7' in ch:
                continue
            if d=='Y7' and 'Y2' in ch:
                continue
            
            for i in installs:
                models = table.query('INSTALL==\'' + i + '\' and (SLED==\'new_accel\' or SLED==\'' + c + '\')')['MODEL'].unique()
                
                for m in models:
                    subset = table.query('INSTALL==\'' + i + '\' and MODEL==\'' + m + '\' and (SLED==\'new_accel\' or SLED==\'' + c + '\')')
                    
                    i_new = subset.query('SLED==\'new_accel\'').index
                    i_old = subset.query('SLED==\''+c+'\'').index
                    
                    print(stattable.loc[i_new,'Max_'+ch].mean() - stattable.loc[i_old,'Max_'+ch].mean())
                    
                    if len(subset['SLED'].unique())==1:
                        continue
                    if chdata[ch][subset.index].apply(is_all_nan).all():
                        continue
                    
                    fig, ax = plt.subplots()
                    for test in subset.index:
                        if subset.at[test,'SLED']=='new_accel':
                            plot_colour = 'IndianRed'
                        elif subset.at[test,'SLED']=='old_accel':
                            plot_colour = 'SkyBlue'
                        else:
                            plot_colour = 'g'
                        if ch!='Angle' and not('DUp' in ch) and not('DDown' in ch):
                            ax.plot(t, chdata[ch][test],color=plot_colour,label=test + ' ' + subset.at[test,'SLED'])
                        else:
                            if test in angle_t.index:
                                ax.plot(angle_t[test], chdata[ch][test],color=plot_colour,label=test + ' ' + subset.at[test,'SLED'])
                    ax.set_title(ch + ' ' + m + ' ' + i)
                    plt.legend(bbox_to_anchor=(1,1))
                    plt.show()
                    plt.close(fig)
#%% plot seat trajectories in B11 and B12 installations
models = ['Evenflo Embrace  CRABI','SNUGRIDE CRABI','Cybex Aton Q']
sleds = ['new_accel','old_accel','new_decel']
for c in sleds:
    for m in models:
        subset = table.query('MODEL==\'' + m + '\' and (INSTALL==\'B11\' or INSTALL==\'B12\') and SLED==\''+c+'\'')
        fig, ax = plt.subplots()
        lines = {}
        for se in subset.index:
            if subset.at[se,'SLED']=='new_accel':
                plot_colour = 'IndianRed'
                name2 = ' (New Bench)'
            elif subset.at[se,'SLED']=='old_accel':
                plot_colour = 'SkyBlue'
                name2 = ' (Old Bench)'
            elif subset.at[se,'SLED']=='new_decel':
                plot_colour='g'
                name2 = ' (Deceleration Sled)'
                
            if subset.at[se,'INSTALL']=='B11':
                line = '-'
                name1 = 'Type 2 Belt'
            elif subset.at[se,'INSTALL']=='B12':
                line = '--'
                name1 = 'UAS'
                
            l = ax.plot(-chdata.at[se,'DUp_x'],chdata.at[se,'DUp_y'],line,color=plot_colour,label=name1)
            lines[l[0]._label] = l[0]
        ax.set_title(m.rstrip(' CRABI') + name2)
        ax.legend(handles=lines.values())
        ax.set_xlim([0,220])
        ax.set_ylim([-220,0])
        ax.set_xlabel('Excursion [mm]')
        ax.set_ylabel('V. Displacement [mm]')
        ax = adjust_font_sizes(ax,{'ticklabels':18,'title':20,'axlabels':18,'legend':18})
        plt.show()
        plt.close(fig)
#%% find which comparisons have p<0.05 and beta>threshold 
with open(directory+'params.json','r') as json_file:
    to_JSON = json.load(json_file)
    
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
            is_sig = p.loc[i]<0.05
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

#%% regression
i = table.query('INSTALL==\'HB\'').index

#ch0_list = ['Head_3ms',
#            'Chest_3ms']
ch0_list = ['Min_12NECKUP00Y7FOXA']
#plot_channels = ['Max_12HEAD0000Y2ACRA',
#                 'Max_12CHST0000Y2ACRC',
#                 'Head_3ms',
#                 'Chest_3ms']
#plot_channels = ['Max_Angle',
#                 'Min_DUp_x',
#                 'Min_DUp_y',
#                 'Min_DDown_x',
#                 'Min_DDown_y']
#plot_channels = ['Min_12PELV0000Y7ACXA',
#                 'Min_12PELV0000Y7ACZA']
plot_channels = ['Max_12NECKUP00Y7FOZA']
#plot_channels = [ch for ch in stattable.columns if not 'Y2' in ch and not 'ACR' in ch and not '3ms' in ch]

#i_old = table.query('DUMMY==\'Y7\' and (INSTALL==\'HB\' or INSTALL==\'LB\') and SLED==\'old_accel\'').index
#i_new = table.query('DUMMY==\'Y7\' and (INSTALL==\'HB\' or INSTALL==\'LB\') and SLED==\'new_accel\'').index
#i_decel = table.query('DUMMY==\'Y7\' and (INSTALL==\'HB\' or INSTALL==\'LB\') and SLED==\'new_decel\'').index
i_old = table.query('INSTALL==\'HB\' and SLED==\'old_accel\'').index
i_new = table.query('INSTALL==\'HB\' and SLED==\'new_accel\'').index
i_decel = table.query('INSTALL==\'HB\' and SLED==\'new_decel\'').index
grp = {'old':i_old,'new':i_new,'decel':i_decel}

for ch0 in ch0_list:
    for ch in plot_channels:
        if ch0==ch:
            continue
        
        list_rsq = []
        fig, ax = plt.subplots()
        for i in grp:
            x = stattable.loc[grp[i],ch]
            y = stattable.loc[grp[i],ch0]
        
            drop = x.index[np.append(np.where(np.isnan(x)),np.where(np.isnan(y)))]
        
            x = x.drop(drop)
            y = y.drop(drop)
                
            if len(x)==0 or len(y)==0:
                continue
        
            rsq = float(r2(x,y))
            list_rsq.append(rsq)
        
#            if rsq < 0.1:
#                continue
            
            if i=='old':
                colour = 'SkyBlue'
                label='old'
            elif i=='new':
                colour='IndianRed'
                label='new'
            else:
                colour = 'g'
                label = 'decel'
            ax.plot(x,y,'.',color=colour,label=label+' R2=' + str(rsq)[:5])
        if len(list_rsq)==0:
            plt.close(fig)
            continue
        elif max(list_rsq)<0.0:
            plt.close(fig)
            continue
        
        ax.set_xlabel(ch)
        ax.set_ylabel(ch0)
        ax.legend()
        plt.show()
        plt.close(fig)

#%% get euclidean distance between configurations
subset = table.query('DUMMY==\'Y2\' and INSTALL!=\'H1\' and INSTALL!=\'B22\' and SLED!=\'new_decel\'')
subset_ch = ['DUp_x','DDown_x','DUp_y','DDown_y']
subset_chdata = chdata.loc[subset.index,subset_ch]
drop = subset_chdata['DUp_x'][subset_chdata['DUp_x'].apply(is_all_nan)].index
subset_chdata = subset_chdata.drop(drop)
subset = subset.drop(drop)

min_len = subset_chdata.applymap(len).min().min()
subset_chdata = subset_chdata.applymap(lambda x: x[:min_len])

distances = {i: {} for i in subset['INSTALL'].unique()}

for i in subset['INSTALL'].unique():
    models = subset.query('INSTALL==\''+i+'\'')['MODEL'].unique()
    for m in models:
        i_old = subset.query('INSTALL==\''+i+'\' and MODEL==\''+m+'\' and SLED==\'old_accel\'').index
        i_new = subset.query('INSTALL==\''+i+'\' and MODEL==\''+m+'\' and SLED==\'new_accel\'').index
        
        if len(i_old)==0 or len(i_new)==0:
            continue
        
        x_old = subset_chdata.loc[i_old,subset_ch]
        x_new = subset_chdata.loc[i_new,subset_ch]
        
        if len(x_old)>1:
            x_old = x_old.apply(lambda x: np.mean(np.vstack(x),axis=0))
        else:
            x_old = x_old.apply(lambda x: x.values[0])
            
        if len(x_new)>1:
            x_new = x_new.apply(lambda x: np.mean(np.vstack(x),axis=0))
        else:
            x_new = x_new.apply(lambda x: x.values[0])
        
        dist = x_old.sub(x_new).applymap(lambda x: x**2).sum().sum()
        distances[i][m] = dist
    distances[i] = pd.Series(distances[i])

mean_distances = {i: distances[i].mean() for i in distances}
std_distances = {i: distances[i].std() for i in distances}

fig, ax = plt.subplots()
ax.bar(mean_distances.keys(),
       np.asarray(list(mean_distances.values()))/max(mean_distances.values()),
       yerr=np.asarray(list(std_distances.values()))/max(mean_distances.values()),
       capsize=6,
       error_kw={'elinewidth':2,'capthick':2},
       color=(0.65,0.65,0.65))
ax.set_ylabel('Relative Distance')
ax.set_title('(Title TBD)')
ax = adjust_font_sizes(ax,{'ticklabels':18,'title':20,'ylabel':18})

#%% euclidean distance of B11 vs B12
models = ['Evenflo Embrace  CRABI','SNUGRIDE CRABI','Cybex Aton Q']
sleds = ['new_accel','old_accel','new_decel']
indices = np.arange(len(models))
distances = {i: {} for i in comparison}
subset_ch = ['DUp_x','DDown_x','DUp_y','DDown_y']
subset = query_list(query_list(table,'MODEL',models),'INSTALL',['B11','B12'])
subset_chdata = chdata.loc[subset.index,subset_ch]
drop = subset_chdata['DUp_x'][subset_chdata['DUp_x'].apply(is_all_nan)].index
subset_chdata = subset_chdata.drop(drop)
subset = subset.drop(drop)

min_len = subset_chdata.applymap(len).min().min()
subset_chdata = subset_chdata.applymap(lambda x: x[:min_len])

distances = pd.DataFrame(index=sleds,columns=models)
for c in sleds:
    for m in models:
        i_b11 = subset.query('SLED==\'' + c + '\' and INSTALL==\'B11\' and MODEL==\'' + m + '\'').index
        i_b12 = subset.query('SLED==\'' + c + '\' and INSTALL==\'B12\' and MODEL==\'' + m + '\'').index
        
        x_b11 = subset_chdata.loc[i_b11,subset_ch]
        x_b12 = subset_chdata.loc[i_b12,subset_ch]
        
        if len(x_b11)>1:
            x_b11 = x_b11.apply(lambda x: np.mean(np.vstack(x),axis=0))
        else:
            x_b11 = x_b11.apply(lambda x: x.values[0])     
            
        if len(x_b12)>1:
            x_b12 = x_b12.apply(lambda x: np.mean(np.vstack(x),axis=0))
        else:
            x_b12 = x_b12.apply(lambda x: x.values[0])
            
        dist = x_b11.sub(x_b12).applymap(lambda x: x**2).sum().sum()
        distances.at[c,m] = dist

for c in comparison:
    fig, ax = plt.subplots()
    max_dist = distances.loc[['new_accel',c]].max().max()
    ax.bar(indices-width/2,distances.loc['new_accel']/max_dist,width,color='IndianRed',label='New Bench')
    if c=='old_accel':
        plot_colour = 'SkyBlue'
        name = 'Old Bench'
    else:
        plot_colour = '#589F58'
        name = 'Deceleration Sled'
    ax.bar(indices+width/2,distances.loc[c]/max_dist,width,color=plot_colour,label=name)
    ax.set_xticks(indices)
    ax.set_xticklabels([i.rstrip(' CRABI').replace(' ','\n') for i in distances.columns])
    ax.legend(bbox_to_anchor=(1.1,0.6))
    ax.set_ylabel('Relative Distance')
    ax.set_title('(Title TBD)')
    ax = adjust_font_sizes(ax,{'ticklabels': 18,'title': 20,'legend':18,'ylabel':18})

#%% plot overlay
from PMG.COM.plotfuns import plot_overlay
#plot_channels = [ch for ch in chdata.columns if not 'Y2' in ch and not 'ACR' in ch and not '3ms' in ch]
plot_channels = ['12NECKUP00Y7FOXA',
                 '12NECKUP00Y7FOZA']
i_old = table.query('INSTALL==\'HB\' and SLED==\'old_accel\'').index
i_new = table.query('INSTALL==\'HB\' and SLED==\'new_accel\'').index

for ch in plot_channels:
    if chdata.loc[i_old,ch].apply(is_all_nan).all() or chdata.loc[i_new,ch].apply(is_all_nan).all():
        continue
    
    fig, ax = plt.subplots()
#    ax = plot_overlay(ax,angle_t,[chdata.loc[i_old,ch].values,'old'],[chdata.loc[i_new,ch].values,'new'])
    ax = plot_overlay(ax,t,[chdata.loc[i_old,ch].values,'old'],[chdata.loc[i_new,ch].values,'new'])
    ax.set_title(ch)
    ax.legend(bbox_to_anchor=(1.1,1))
#    ax.set_ylim([-50,50])
    plt.show()
    plt.close(fig)