# -*- coding: utf-8 -*-
"""
LAB BELT SCORE PLOT
    Compare the lap belt score to key data to establish any correlations

Created on Fri Nov 10 11:43:05 2017

@author: giguerf
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
if 'C:/Users/giguerf/Documents' not in sys.path:
    sys.path.insert(0, 'C:/Users/giguerf/Documents')
from GitHub.COM import openbook as ob
#from GitHub.COM import plotstyle as style
from GitHub.COM import get_peak as gp

readdir = os.fspath('P:/BOOSTER/SAI/Y7')
savedir = os.fspath('P:/BOOSTER/Plots/Y7')

#time, fulldata, *_ = ob.openbook(readdir)

#%%

table = pd.read_excel('P:/BOOSTER/boostertable.xlsx', index_col = 0)
import xlrd
faro = pd.DataFrame(columns = ['LBS'])
for file in os.listdir('P:/BOOSTER/Faro/'):
    tcn = file[:-4]
    for sheetname in ['BELT MEASURE POSITION 14','BELT MEASURE POSITION 16']:
        try:
            raw = pd.read_excel('P:/BOOSTER/Faro/'+tcn+'.xls', sheetname=sheetname, index_col=4)
        except xlrd.XLRDError:
            try:
                raw = pd.read_excel('P:/BOOSTER/Faro/'+tcn+'.xls', sheetname=sheetname.title(), index_col=4)
            except xlrd.XLRDError:
                pass
            else:
                raw = raw['X']
                relative = (raw.iloc[0]-raw).loc[['Top Lap Belt Left','Top Lap Belt Right']]
                faro.loc[tcn+'_'+sheetname[-2:]] = relative.mean(axis=0)
        else:
            raw = raw['X']
            relative = (raw.iloc[0]-raw).loc[['Top Lap Belt Left','Top Lap Belt Right']]
            faro.loc[tcn+'_'+sheetname[-2:]] = relative.mean(axis=0)
temp = pd.merge(table, faro, how = 'outer', left_index = True, right_index = True)
tcnlist = list(table.index)
peakdata = {}
for channel in fulldata:
    peakdata[channel] = pd.DataFrame(columns = ['t0','tp','start','peak'])
    for tcn in fulldata[channel].columns.intersection(tcnlist):
        if channel in ['14ILACRIUPY7FOXB','14ILACRILOY7FOXB','14ILACLEUPY7FOXB','14ILACLELOY7FOXB']:
            lim = 'max'
        else:
            lim = 'min'
        peakdata[channel].loc[tcn] = gp.get_peak(time, fulldata[channel][tcn], lim=lim)
#%%
peaktable = pd.concat([peakdata['14CHST0000Y7ACXC'], peakdata['14PELV0000Y7ACXA']], axis=1)
peaktable.columns = ['Chest Start Time', 'Chest Peak Time', 'Chest 5% Peak', 'Chest Peak','Pelvis Start Time', 'Pelvis Peak Time', 'Pelvis 5% Peak', 'Pelvis Peak']
t = pd.merge(temp, peaktable, how='outer', left_index=True, right_index=True)
t = t[t['Type_Mannequin'] == 'HIII Enfant'].drop(['Location','Date_essai','Type','Type_Mannequin','Seat'], axis=1)
t = t.reset_index()
#t.to_excel('C:/Users/giguerf/table.xlsx', index=False)
TCNs = ['TC16-129', 'TC14-503_2', 'TC14-503_4', 'TC14-503_5', 'TC16-132', 'TC18-105', 'TC12-004']
datatable = t[t.TCN.isin(TCNs)]
datatable.loc[:,'Delta g'] = -(datatable.loc[:,'Pelvis Peak'] - datatable.loc[:,'Chest Peak'])

datatable.loc[(datatable.Position == 14) & (datatable.ModeleAuto == 'PRIUS'), 'Group'] = 'Lap'
datatable.loc[(datatable.Position == 16) & (datatable.ModeleAuto == 'PRIUS'), 'Group'] = 'Pelvis'
datatable.loc[(datatable.Position == 14) & (datatable.ModeleAuto != 'PRIUS'), 'Group'] = 'Pelvis'
datatable.loc[(datatable.Position == 16) & (datatable.ModeleAuto != 'PRIUS'), 'Group'] = 'Lap'

datatable = datatable.sort_values(['Vitesse'])
datatable = datatable.reset_index(drop=True)

tableout = datatable.loc[:,['TCN','Position','Vitesse', 'Marque', 'Modele','LBS','Group','Delta g']]
#tableout.to_excel('P:/BOOSTER/Plots/barchart.xlsx', index=False)
tablereport = datatable.drop(['index', 'MarqueAuto', 'ModeleAuto', 'ID', 'Marque', 'Modele', 'Group', 'Chest Start Time', 'Chest 5% Peak', 'Pelvis Start Time', 'Pelvis 5% Peak'], axis=1)
tablereport['Delta t'] = -(tablereport.loc[:,'Pelvis Peak Time'] - tablereport.loc[:,'Chest Peak Time'])
#tablereport.to_excel('P:/BOOSTER/Plots/Table3.xlsx', index=False)
#%% Chest / Pelvis
plt.close('all')
#tableout = datatable.loc[:,['TCN','Position','Vitesse', 'Marque', 'Modele','LBS','Group','Delta g']]
tableout=tableout.rename(columns = {'Delta g':'Toward'})
df1 = tableout[tableout.Group == 'Lap'].drop(['Vitesse','Group'], axis = 1)
df2 = tableout[tableout.Group == 'Pelvis'].drop(['Vitesse','Group'], axis = 1)
df = pd.merge(df1,df2,how='inner',on='TCN',suffixes=[' Lap',' Pelvis'])
df = df.set_index('TCN')

#df.loc[:,'Disp']  = df.loc[:,'Toward Lap']-df.loc[:,'Toward Pelvis']
#df = df.sort_values(['Disp'])
#df2 = pd.DataFrame.copy(df)
#df.loc['TC18-105',['Toward Pelvis','Toward Lap','Disp']] = 0
#df2.loc[list(set(df.index) - set(['TC18-105'])),['Toward Pelvis','Toward Lap','Disp']] = 0
ax = df[['Toward Pelvis','Toward Lap']].plot(kind='barh', width=0.75)
#ax2 = df2[['Toward Pelvis','Toward Lap']].plot(kind='barh', width=0.75, color = [(0.122,0.467,0.701,0.7),(1,0.498,0.055,0.7)], ax=ax)

#for i, each in enumerate(df.index):
#    for col in ['LBS Lap','LBS Pelvis']:
#        y = df.loc[each][col]
#        ax.text(-7.5, i+0.1, np.round(y, 1), fontsize=8,horizontalalignment='center',verticalalignment='center')
#        i=i-0.25
for i, each in enumerate(df.index):
    y = df.loc[each]['LBS Lap']-df.loc[each]['LBS Pelvis']
    ax.text(-7.5, i+0.1, np.round(y, 1), fontsize=8,horizontalalignment='center',verticalalignment='center')
    if each == 'TC18-105':
        ax.text(df.loc['TC18-105']['Toward Lap']+1, i+0.1, '56 km/h', fontsize=8,horizontalalignment='left',verticalalignment='center')

plt.xticks(np.arange(-10,40,5))
plt.setp(ax.get_yticklabels(), visible=False)

ax.set_xlim((-10,40))
ax.set_title('In-Vehicle Paired Comparison')
ax.set_xlabel('Disparity between Chest and Pelvis in Peak Acceleration (X-direction) [g]\n\nWhen value is positive, the Pelvis Peak is greater in magnitude\n than the Chest Peak')
ax.set_ylabel('Paired tests')
h, l = ax.get_legend_handles_labels()
ax.legend([h[1][0],h[0][0]], ['Belt '+l[1],'Belt '+l[0]], loc = 5, bbox_to_anchor=(1, 0.65))
plt.tight_layout()
ax.set_axisbelow(True)
ax.grid(color = '0.80')

plt.savefig('P:/BOOSTER/Plots/barchart.png', dpi = 300, bbox_inches="tight")
#df.to_excel('P:/BOOSTER/Plots/barchart2.xlsx')
#%%
peaktable = pd.concat([peakdata['14LUSP0000Y7FOXA'], peakdata['14LUSP0000Y7FOZA']], axis=1)
peaktable.columns = ['Lumbar X Start Time', 'Lumbar X Peak Time', 'Lumbar X 5% Peak', 'Lumbar X Peak','Lumbar Z Start Time', 'Lumbar Z Peak Time', 'Lumbar Z 5% Peak', 'Lumbar Z Peak']
t = pd.merge(temp, peaktable, how='outer', left_index=True, right_index=True)
t = t[t['Type_Mannequin'] == 'HIII Enfant'].drop(['Location','Date_essai','Type','Type_Mannequin','Seat'], axis=1)
t = t.reset_index()
#t.to_excel('C:/Users/giguerf/table.xlsx', index=False)
TCNs = ['TC12-004','TC16-129','TC16-132','TC18-105',
        'TC14-503_2','TC14-503_4','TC14-503_5']
datatable = t[t.TCN.isin(TCNs)]
datatable.loc[:,'Toward'] = datatable.loc[:,'Lumbar X Peak'] - datatable.loc[:,'Lumbar Z Peak']

datatable.loc[(datatable.Position == 14) & (datatable.ModeleAuto == 'PRIUS'), 'Group'] = 'Lap'
datatable.loc[(datatable.Position == 16) & (datatable.ModeleAuto == 'PRIUS'), 'Group'] = 'Pelvis'
datatable.loc[(datatable.Position == 14) & (datatable.ModeleAuto != 'PRIUS'), 'Group'] = 'Pelvis'
datatable.loc[(datatable.Position == 16) & (datatable.ModeleAuto != 'PRIUS'), 'Group'] = 'Lap'

datatable = datatable.sort_values(['Vitesse'])
datatable = datatable.reset_index(drop=True)
#%% Lumbar X / Z
#plt.close('all')
tableout = datatable.loc[:,['TCN','Position','Vitesse', 'Marque', 'Modele','LBS','Group','Toward']]

df1 = tableout[tableout.Group == 'Lap'].drop(['Vitesse','Group'], axis = 1)
df2 = tableout[tableout.Group == 'Pelvis'].drop(['Vitesse','Group'], axis = 1)
df = pd.merge(df1,df2,how='inner',on='TCN',suffixes=[' Lap',' Pelvis'])
df = df.set_index('TCN')

#for dim in ['X','Z']:
#    for tcn in df.index:
#        lapx = int(datatable[(datatable.TCN == tcn) & (datatable.Group == 'Lap')]['Lumbar '+dim+' Peak'])
#        pelx = int(datatable[(datatable.TCN == tcn) & (datatable.Group == 'Pelvis')]['Lumbar '+dim+' Peak'])
#        df.loc[tcn,'L'+dim+' Gap'] = pelx-lapx
#df['Disp']  = df.loc[:,'Toward Lap']-df.loc[:,'Toward Pelvis']
#df = df.sort_values(['Disp'])

ax = df[['Toward Pelvis','Toward Lap',]].plot(kind='barh', width=0.75)
#ax2 = df2[['Toward Pelvis','Toward Lap']].plot(kind='barh', width=0.75, color = [(0.122,0.467,0.701,0.7),(1,0.498,0.055,0.7)], ax=ax)

#for i, each in enumerate(df.index):
#    for col in ['LBS Lap','LBS Pelvis']:
#        y = df.loc[each][col]
#        ax.text(-7.5, i+0.1, np.round(y, 1), fontsize=8,horizontalalignment='center',verticalalignment='center')
#        i=i-0.25
for i, each in enumerate(df.index):
    y = df.loc[each]['LBS Lap']-df.loc[each]['LBS Pelvis']
    ax.text(-750, i+0.1, np.round(y, 1), fontsize=8,horizontalalignment='center',verticalalignment='center')
    if each == 'TC18-105':
        ax.text(100, i, '56 km/h', fontsize=8,horizontalalignment='left',verticalalignment='center')

plt.xticks(np.arange(-1000,1250,250))
plt.setp(ax.get_yticklabels(), visible=False)

ax.set_xlim((-1000,1250))
ax.set_title('In-Vehicle Paired Comparison')
ax.set_xlabel('Disparity between Lumbar Spine X and Z Peak Forces [N]\n\nWhen value is positive, the Lumbar Z Peak is greater in magnitude\n than the Lumbar X Peak')
ax.set_ylabel('Paired tests')
h, l = ax.get_legend_handles_labels()
#ax.legend([h[1][0],h[0][0]], ['Lumbar Z Gap','Lumbar X Gap'], loc = 5, bbox_to_anchor=(1, 0.75))
ax.legend([h[1][0],h[0][0]], ['Belt '+l[1],'Belt '+l[0]], loc = 5, bbox_to_anchor=(1, 0.55))
plt.tight_layout()
ax.set_axisbelow(True)
ax.grid(color = '0.80')

plt.savefig('P:/BOOSTER/Plots/barchart_Lumbar.png', dpi = 300, bbox_inches="tight")
#%%
peakright = pd.concat([peakdata['14ILACRIUPY7FOXB'], peakdata['14ILACRILOY7FOXB']], axis=1)
peakleft = pd.concat([peakdata['14ILACLEUPY7FOXB'], peakdata['14ILACLELOY7FOXB']], axis=1)
peaktable = pd.concat([peakleft,peakright], axis=0, join='outer')
#peaktable = pd.concat([peakdata['14ILACLEUPY7FOXB'], peakdata['14ILACLELOY7FOXB']], axis=1)
peaktable.columns = ['Il Up Start Time', 'Il Up Peak Time', 'Il Up 5% Peak', 'Il Up Peak','Il Lo Start Time', 'Il Lo Peak Time', 'Il Lo 5% Peak', 'Il Lo Peak']
t = pd.merge(temp, peaktable, how='outer', left_index=True, right_index=True)
t = t[t['Type_Mannequin'] == 'HIII Enfant'].drop(['Location','Date_essai','Type','Type_Mannequin','Seat'], axis=1)
t = t.reset_index()
#t.to_excel('C:/Users/giguerf/table.xlsx', index=False)
TCNs = ['TC12-004','TC16-129','TC16-132','TC18-105',
        'TC14-503_2','TC14-503_4','TC14-503_5']
datatable = t[t.TCN.isin(TCNs)]
datatable.loc[:,'Toward'] = -(datatable.loc[:,'Il Up Peak'] - datatable.loc[:,'Il Lo Peak'])

datatable.loc[(datatable.Position == 14) & (datatable.ModeleAuto == 'PRIUS'), 'Group'] = 'Lap'
datatable.loc[(datatable.Position == 16) & (datatable.ModeleAuto == 'PRIUS'), 'Group'] = 'Pelvis'
datatable.loc[(datatable.Position == 14) & (datatable.ModeleAuto != 'PRIUS'), 'Group'] = 'Pelvis'
datatable.loc[(datatable.Position == 16) & (datatable.ModeleAuto != 'PRIUS'), 'Group'] = 'Lap'

datatable = datatable.sort_values(['Vitesse'])
datatable = datatable.reset_index(drop=True)
#%% Iliac Upper/Lower
#plt.close('all')
tableout = datatable.loc[:,['TCN','Position','Vitesse', 'Marque', 'Modele','LBS','Group','Toward']]

df1 = tableout[tableout.Group == 'Lap'].drop(['Vitesse','Group'], axis = 1)
df2 = tableout[tableout.Group == 'Pelvis'].drop(['Vitesse','Group'], axis = 1)
df = pd.merge(df1,df2,how='inner',on='TCN',suffixes=[' Lap',' Pelvis'])
df = df.set_index('TCN')

#for dim in ['X','Z']:
#    for tcn in df.index:
#        lapx = int(datatable[(datatable.TCN == tcn) & (datatable.Group == 'Lap')]['Lumbar '+dim+' Peak'])
#        pelx = int(datatable[(datatable.TCN == tcn) & (datatable.Group == 'Pelvis')]['Lumbar '+dim+' Peak'])
#        df.loc[tcn,'L'+dim+' Gap'] = pelx-lapx
#df['Disp']  = df.loc[:,'Toward Lap']-df.loc[:,'Toward Pelvis']
#df = df.sort_values(['Disp'])

ax = df[['Toward Pelvis','Toward Lap',]].plot(kind='barh', width=0.75)
#ax2 = df2[['Toward Pelvis','Toward Lap']].plot(kind='barh', width=0.75, color = [(0.122,0.467,0.701,0.7),(1,0.498,0.055,0.7)], ax=ax)

#for i, each in enumerate(df.index):
#    for col in ['LBS Lap','LBS Pelvis']:
#        y = df.loc[each][col]
#        ax.text(-7.5, i+0.1, np.round(y, 1), fontsize=8,horizontalalignment='center',verticalalignment='center')
#        i=i-0.25
for i, each in enumerate(df.index):
    y = df.loc[each]['LBS Lap']-df.loc[each]['LBS Pelvis']
    ax.text(-200, i+0.1, np.round(y, 1), fontsize=8,horizontalalignment='center',verticalalignment='center')
    if each == 'TC18-105':
        ax.text(900, i, '56 km/h', fontsize=8,horizontalalignment='left',verticalalignment='center')

plt.xticks(np.arange(-400,1200,200))
plt.setp(ax.get_yticklabels(), visible=False)

ax.set_xlim((-400,1200))
ax.set_title('In-Vehicle Paired Comparison')
ax.set_xlabel('Disparity between Illiac Spine Upper and Lower Peak Forces [N]\n\nWhen value is positive, the Iliac Lower Peak is greater in magnitude\n than the Iliac Upper Peak')
ax.set_ylabel('Paired tests')
h, l = ax.get_legend_handles_labels()
#ax.legend([h[1][0],h[0][0]], ['Lumbar Z Gap','Lumbar X Gap'], loc = 5, bbox_to_anchor=(1, 0.75))
ax.legend([h[1][0],h[0][0]], ['Belt '+l[1],'Belt '+l[0]], loc = 5, bbox_to_anchor=(1, 0.45))
plt.tight_layout()
ax.set_axisbelow(True)
ax.grid(color = '0.80')

plt.savefig('P:/BOOSTER/Plots/barchart_Iliac.png', dpi = 300, bbox_inches="tight")
#%% Correlate lbs with Chest/Pelvis Gap
#from GitHub.COM import plotstyle as style
#plt.close('all')
peaktable = pd.concat([peakdata['14CHST0000Y7ACXC'], peakdata['14PELV0000Y7ACXA']], axis=1)
peaktable.columns = ['Chest Start Time', 'Chest Peak Time', 'Chest 5% Peak', 'Chest Peak','Pelvis Start Time', 'Pelvis Peak Time', 'Pelvis 5% Peak', 'Pelvis Peak']
t = pd.merge(temp, peaktable, how='outer', left_index=True, right_index=True)
t = t[t['Type_Mannequin'] == 'HIII Enfant'].drop(['Location','Date_essai','Type','Type_Mannequin','Seat'], axis=1)
t = t.reset_index()
#t.to_excel('C:/Users/giguerf/table.xlsx', index=False)
TCNs = ['TC12-004','TC16-129','TC16-132','TC18-105',
        'TC14-503_2','TC14-503_4','TC14-503_5']
datatable = t[t.TCN.isin(TCNs)]
datatable.loc[:,'Delta'] = -(datatable.loc[:,'Pelvis Peak'] - datatable.loc[:,'Chest Peak'])

datatable.loc[(datatable.Position == 14) & (datatable.ModeleAuto == 'PRIUS'), 'Group'] = 'Lap'
datatable.loc[(datatable.Position == 16) & (datatable.ModeleAuto == 'PRIUS'), 'Group'] = 'Pelvis'
datatable.loc[(datatable.Position == 14) & (datatable.ModeleAuto != 'PRIUS'), 'Group'] = 'Pelvis'
datatable.loc[(datatable.Position == 16) & (datatable.ModeleAuto != 'PRIUS'), 'Group'] = 'Lap'

datatable = datatable.sort_values(['Vitesse'])
datatable = datatable.reset_index(drop=True)

tableout = datatable.loc[:,['TCN','Position','Vitesse', 'Marque', 'Modele','LBS','Group','Delta']]
#tableout.to_excel('P:/BOOSTER/Plots/barchart.xlsx', index=False)

plt.figure()
ax = plt.gca()
#ax.plot(tableout['LBS'],tableout['Delta'],'.')
#tableout.loc[:,'color'] = 0
#colors = style.colordict(['312-NEO JUNIOR SEAT', 'AFFIX', 'AMP', 'BOOSTER SEAT', 'MF01-US', 'TURBO BOOSTER', 'VIAGGIO HBB'])
#for tcn in tableout.index:
#    tableout.loc[tcn,'color'] = colors[tableout.loc[tcn,'Modele']]
#    ax.plot([float('nan')],[float('nan')], color = colors[tableout.loc[tcn,'Modele']], label = tableout.loc[tcn,'Modele'].title())

tableout.plot.scatter('LBS', 'Delta', ax=ax, s=10)#, c=tableout['color'])
m, b, r, p, sig = scipy.stats.linregress(tableout['LBS'], tableout['Delta'])
#fit = np.polyfit(tableout['LBS'],tableout['Delta'],1)
fit_fn = np.poly1d([m,b])
ax.plot(tableout['LBS'], fit_fn(tableout['LBS']), color = (.60,.60,.60))
ax.text(-10, fit_fn(-10)+1, '%s\n$R^2$ = %s' % (fit_fn, np.round(r**2,2)), rotation = 23)

ax.set_xlabel('Lap Belt Score (Distance from Pubis in X-Direction)')
ax.set_ylabel('Disparity between Chest and Pelvis Peak Accelerations [g]')

#h, l = ax.get_legend_handles_labels()
#ax.legend(h, l, loc = 5, bbox_to_anchor=(1, 0.45))
#style.legend(ax, loc = 'lower right')

plt.savefig('P:/BOOSTER/Plots/scatter.png', dpi = 300, bbox_inches="tight")
